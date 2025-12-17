import os

import torch
import warnings
from .model_minimind import *
from typing import Optional, Tuple, List
from torch import nn
from transformers import CLIPProcessor, CLIPModel
from typing import List

warnings.filterwarnings('ignore')


class VLMConfig(MiniMindConfig):
    model_type = "minimind-v"

    def __init__(
            self,
            # 图像在文本中的占位符信息 ： image_special_token: 196 个 @ 符号
            image_special_token: str = '@' * 196,
            # 196 ： CLIP (ViT-B/16) 处理 $224 \times 224$ 的图片时，会将其切分为 $14 \times 14 = 196$ 个 patch（图块），每个 patch 对应一个特征向量
            # image_ids: 对应的 token ID 列表（假设 ID 为 34），也就是tokenizer会把 @ 映射为 34
            image_ids: List = [34] * 196, 
            **kwargs,
    ):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        super().__init__(**kwargs)

class VisionProj(nn.Module):
    def __init__(self, ve_hidden_size=768, hidden_size=512):
        super().__init__()
        # 一个简单的全连接层
        # 维度对齐。CLIP 输出的特征维度是 ve_hidden_size (768)，而 MiniMind 语言模型的维度是 hidden_size (512)
        self.ve_hidden_size = ve_hidden_size
        self.hidden_size = hidden_size
        # Sequential 意思是“按顺序”。它把放入其中的网络层串联起来，组成一个大的模块。
        """
        既然只有一层，为什么用Sequential？ 
        因为以后可以扩展，如果以后想升级模型，直接在这里加：
        self.vision_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),           # 加个激活函数
            nn.Linear(512, 512)  # 再加一层
        )
        """
        # Linear(A,B),把A维度映射为B维度
        self.vision_proj = nn.Sequential(
            nn.Linear(self.ve_hidden_size, self.hidden_size)
        )

    def forward(self, image_encoders):
        vision_proj = self.vision_proj(image_encoders)
        return vision_proj


# 继承自语言模型
class MiniMindVLM(MiniMindForCausalLM):
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None, vision_model_path="./model/vision_model/clip-vit-base-patch16"):
        # 把语言模型原本的（Transformer层、Embedding层等）先装好
        super().__init__(params)
        if not params: params = VLMConfig()
        self.params = params
        # processor 是处理图像的工具
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        # 初始化投影层（需要训练）
        self.vision_proj = VisionProj(hidden_size=params.hidden_size)

    # 这是一个属于该类的工具函数，但它在这个类的‘命名空间’下运行时，不需要访问类的实例（self）或类本身（cls）
    @staticmethod
    def get_vision_model(model_path: str):

        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        
        # 加载预训练好的 CLIP 模型和处理器
        model = CLIPModel.from_pretrained(model_path)
        processor = CLIPProcessor.from_pretrained(model_path)

        # 遍历模型里所有的参数（权重矩阵、偏置等）
        for param in model.parameters():
            # 将“需要梯度”这个开关设为 False，冻结 vision_encoder 的所有参数
            param.requires_grad = False


        """
        model.eval(): 开启“评估模式”。将模型实例中的 self.training 标志位递归地设置为 False
        主要影响两类对“训练”和“推理”行为定义完全不同的层：Dropout 和 Normalization（主要是 BatchNorm） 

        """
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        # 如果图片是透明底(RGBA)或灰度(LA)，强行转成普通的 RGB 彩色图
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        # processor 是 CLIP 自带的工具。
        # 它会把图片缩放、裁剪、归一化，变成 PyTorch 能读懂的 Tensor（张量/多维数组）。
        # 'pixel_values' 就是最终的数字矩阵。
        """
        第一步：几何变换:先把图片按比例缩放，让短边变成 224,从图片中心切出一块 224 * 224 的正方形
        第二步：数值缩放:把 0-255 的整数除以 255。所有像素值变成了 0.0 到 1.0 之间的小数。神经网络喜欢小数值，大整数会导致梯度爆炸。
        第三步：统计归一化:为了让图片的像素分布符合 CLIP 训练时的统计规律,它会对每个像素点 $x$ 执行标准分公式,(x-mean)/std.std是标准差
        第四步：维度重排:PIL 格式: $(H, W, C)$ -> $(224, 224, 3)$。这是人类习惯，颜色在最后。
                      PyTorch 格式: $(C, H, W)$ -> $(3, 224, 224)$。这是 PyTorch 习惯，颜色通道在前。
                      增加 Batch 维: 变成 $(1, 3, 224, 224)$。
        """
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        # 这里面的计算不需要算梯度（不反向传播），为了省显存和加速
        with torch.no_grad():
            # 把图片数字矩阵喂给 vision_model，得到输出
            outputs = vision_model.vision_model(pixel_values=image_tensors)

        # outputs.last_hidden_state: 模型最后一层的输出，形状通常是 [Batch, 197, 768]
        # [:, 1:, :] 的意思是：
        # 第0个位置通常是分类标签（CLS token），我们不要。
        # 我们只要后面 196 个（1:）代表图像具体内容的特征块（Patch）。
        # .squeeze(): 如果有维度是1的，把它压缩掉（比如 [1, 196, 768] -> [196, 768]）。
        # 这个 .squeeze() 在这里的唯一目的，就是为了处理 Batch Size = 1 的情况，去掉那个“多余”的维度。
        img_embedding = outputs.last_hidden_state[:, 1:, :].squeeze()
        return img_embedding

    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):

        # 寻找 input_ids 里哪里是图片占位符
        def find_indices(tokens, image_ids):
            # 把python列表，变成pytorch的张量（tensor),
            # .to(tokens.device)：如果你的tokens数据在 GPU 上，而你的目标列表还在 CPU 上，它俩是没法比较的。这行代码确保它俩在同一个设备上。
            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
            len_image_ids = len(image_ids)
            if len_image_ids > tokens.size(1):
                return None
            
            # unfold 滑动窗口，拿着 [34, 34...34] 这个列表，在输入的 tokens 里从头滑到尾，找到完全匹配的位置。
            # 第一个1 是沿着1维，第三个1 是步长
            tokens_view = tokens.unfold(1, len_image_ids, 1)
            """
            data = [                 # Dim 0 (Batch层)
                [                    # Dim 1 (窗口层)
                    [True, True, True],   # Dim 2 (细节层): 第1个窗口里的3个匹配结果
                    [True, False, True]   # Dim 2 (细节层): 第2个窗口里的3个匹配结果
                ]
            ]
            .all(dim=2) : 看到 [True, True, True]执行逻辑与（AND）：True & True & True = True结果：这一串变成了一个单一的 True
            最后的结果变成：
            第三维度被捏扁
            result = [
                [True, False] 
            ]
            # 形状变成了 [1, 2]
            """
            matches = (tokens_view == image_ids_tensor).all(dim=2)
            return {
                batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                            matches[batch_idx].nonzero(as_tuple=True)[0]]
                for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
            } or None

        # 占位符的位置
        image_indices = find_indices(tokens, self.params.image_ids)
        # 如果有图像数据，且找到了占位符位置
        if vision_tensors is not None and image_indices:
            # 投影：把图像特征从 768维 变成 512维
            vision_proj = self.vision_proj(vision_tensors)
            if len(vision_proj.shape) == 3:
                # 后续的代码有一个循环：for i in range(h.size(0)):（遍历每一个样本）。 程序期望 vision_proj 的结构是：[第几个样本, 第几张图, 196个块, 512维] (4维)
                # 之前的 squeeze() 把 Batch 维弄丢了，现在要重新插回batch维度
                vision_proj = vision_proj.unsqueeze(0)
            # 替换embedding
            new_h = []
            for i in range(h.size(0)):
                # 参数 h 是 Hidden States（隐藏状态），它是由 input_ids（那串数字）经过模型的 Embedding 层 查表变出来的
                """
                0	看	[0.1, -0.5, ...]	正常文本向量
                1	图	[0.8, 0.2, ...]	    正常文本向量
                2	:	[-0.1, 0.9, ...]	正常文本向量
                3	@	[0.0, 0.0, ...]	    占位符向量 (ID 34 的向量)
                4	@	[0.0, 0.0, ...]	    占位符向量 (ID 34 的向量)
                5	@	[0.0, 0.0, ...]	    占位符向量 (ID 34 的向量)
                6	是	[0.3, -0.3, ...]	正常文本向量
                """
                if i in image_indices:
                    h_i = h[i] # 拿到当前样本的文本 Embedding
                    img_idx = 0
                    # # 遍历找到的所有占位符区间
                    for start_idx, end_idx in image_indices[i]:
                        # torch.cat 是“拼接”操作。
                        # 下面这行代码的意思是：
                        # 新向量 = [前半段文本] + [转换后的图像特征] + [后半段文本]
                        # 就像剪辑磁带一样，把中间那段无意义的占位符剪掉，接上图像特征。
                        if img_idx < vision_proj.size(1):
                            h_i = torch.cat((h_i[:start_idx], vision_proj[i][img_idx], h_i[end_idx + 1:]), dim=0)[
                                  :seqlen]
                            img_idx += 1
                    new_h.append(h_i)
                else:
                    new_h.append(h[i]) # 如果没图，就原样放回去
            """
            位置	对应的字	new_h 里的内容	    状态
            0	    看	        h[看]	        原样保留
            1	    图	        h[图]	        原样保留
            2	    :	        h[:]	        原样保留
            3	    (图)	CLIP特征[Patch1]	已替换！真·视觉信息
            4	    (图)	CLIP特征[Patch2]	已替换！真·视觉信息
            5	    (图)	CLIP特征[Patch3]	已替换！真·视觉信息
            6	    是	        h[是]	        原样保留
            """
            return torch.stack(new_h, dim=0) # stack: 把处理好的一个个样本重新堆叠成一个大矩阵，它负责把一堆形状相同的张量（Tensor），沿着一个新的维度，堆成一个更大的张量。
        return h

    def forward(self,
                input_ids: Optional[torch.Tensor] = None, # 输入的文字ID [batch, seq_len]
                attention_mask: Optional[torch.Tensor] = None, # # 输入的图片像素
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        # 初始化kv cache
        past_key_values = past_key_values or [None] * len(self.model.layers)
        # 推理时：比如你已经生成了 "I like"，现在要生成 "apple"。模型不需要重新算 "I like"，它直接从第 3 个位置开始算。这时 start_pos 就是 2
        # past_key_values[0] 是 （key,value),past_key_values[0][0]是key。.shape[1]：取出第 1 个维度的大小，第 1 维通常代表 Sequence Length（序列长度）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 1. Embedding（向量化）
        # self.model.embed_tokens(input_ids): 
        # 把输入的数字ID（如 [101, 34, 34...]）查表变成向量。
        # 此时 hidden_states 全是文本向量，其中 34 对应的只是无意义的初始向量。
        # drop_out 每次训练都会扔一些，模拟噪声，破坏对特定token的过度依赖
        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        # 2. 视觉注入（关键！）
        """
        【实战案例解析:Batch=2, 每个样本2张图】
        假设场景：
          Sample 0 (Batch 0): "左边是猫[图1]，右边是狗[图2]"
          Sample 1 (Batch 1): "这是苹果[图3]，那是香蕉[图4]"
        
        1. 初始输入 pixel_values:
           形状: [2, 2, 3, 224, 224]  -> (Batch=2, Num_Images=2, C=3, H=224, W=224)
        
        2. 循环提取特征 (for i in range(num)):
           ------------------------------------------------------
           [第 1 轮 (i=0)] 处理所有人的"第1张图" (猫 & 苹果)
           -切片 input: pixel_values[:, 0, ...] -> 形状 [2, 3, 224, 224]
           -CLIP output: 得到 Tensor_A          -> 形状 [2, 196, 768]
           ------------------------------------------------------
           [第 2 轮 (i=1)] 抽取所有人的"第2张牌" (狗 & 香蕉)
           - 动作解析: pixel_values[:, 1, ...]
             * ":"  代表 "所有样本" (Batch 0 和 Batch 1)
             * "1"  代表 "第2张图"  (Sample 0的狗, Sample 1的香蕉)
        
           - 提取出的 Input (临时Batch):
             [ Sample 0 的狗图 (3x224x224),
               Sample 1 的香蕉图 (3x224x224) ]
             -> 形状变为 [2, 3, 224, 224] (注意: 图片数量维度被切掉了，变成了普通的Batch)
        
           - CLIP 编码 Output:
             CLIP 同时看这两张图，分别计算出特征。
             -> 得到 Tensor_B: 形状 [2, 196, 768] (2个样本, 196个块, 768维)
        
        3. 堆叠 (torch.stack):
           - 操作: stack([Tensor_A, Tensor_B], dim=1)
           - 结果 vision_tensors: 形状 [2, 2, 196, 768]
           - 含义: [Batch, 图片序号, 特征块数, 特征维度]
        
        4. 替换 (count_vision_proj):
           - 对于 Sample 0: 找到 input_ids 里的两个占位符区间，依次填入 Tensor_A[0](猫) 和 Tensor_B[0](狗)
           - 对于 Sample 1: 找到 input_ids 里的两个占位符区间，依次填入 Tensor_A[1](苹果) 和 Tensor_B[1](香蕉)
        
        """
        # 如果传入了图片(pixel_values)，且是刚开始生成(start_pos == 0)
        if pixel_values is not None and start_pos == 0:
            # 处理图片维度，确保是 [batch, num_images, 3, 224, 224]
            if len(pixel_values.shape) == 6: # 有时候因为数据加载器（DataLoader）写法的问题，会莫名其妙多包一层皮。
                pixel_values = pixel_values.squeeze(2)
            bs, num, c, im_h, im_w = pixel_values.shape
            stack_dim = 1 if bs > 1 else 0

            # 提取图片特征：调用之前的 get_image_embeddings
            vision_tensors = torch.stack([
                # 兼容 Batch=1 和 Batch>1 两种情况，pixel_values[:, i, :, :, :] 的意思就是：“保留 Batch 维度（所有样本），只取第 i 张图”
                MiniMindVLM.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)
                for i in range(num)
            ], dim=stack_dim)
            # 调用“手术函数”，把 hidden_states 里的占位符换成真正的图片特征
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors,
                                                   seqlen=input_ids.shape[1])

        # 3. 准备位置编码 (RoPE)
        # Transformer 需要知道“第一个字”和“第二个字”的位置区别。
        # 这里准备了 cos 和 sin 函数用于后续计算位置。
        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 4. Transformer 层循环 (核心思考过程)
        presents = []
        # 遍历每一层 Transformer Layer (self.model.layers)
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            # hidden_states 进去，新的 hidden_states 出来。
            # 这就是深度学习的“深度”所在，一层层提取更高级的语义。
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present) # 存下 KV Cache，为了下次生成加速

        # 5. 归一化 (Norm)
        # 让数据分布更稳定，防止数值过大或过小。
        hidden_states = self.model.norm(hidden_states)

        # 计算辅助损失（Auxiliary Loss）
        # 这只针对 MoE (混合专家) 模型。
        # 如果某个专家太累，某个专家太闲，模型会学偏。这个 Loss 强迫模型均匀使用所有专家。

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.model.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        # 6. 计算 Logits (预测结果)
        # self.lm_head: 这是一个线性层。
        # 它的作用是把 hidden_states (512维) 映射回 词表大小 (比如 32000维)。
        # 每一个维度代表一个词的概率打分。
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        # 7. 打包返回
        output = CausalLMOutputWithPast(logits=logits, past_key_values=presents, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        return output

"""
# ================================================================================================
    【MiniMindVLM 全流程数据流演义】
    
    场景设定：
      Batch Size = 2 (两句话)
      Sample A: "看！[猫图] 和 [狗图]"  (包含2张图，假设每张图占位符长度为 196)
      Sample B: "这是[苹果] 还有 [香蕉]" (包含2张图)
    
    ------------------------------------------------------------------------------------------------
    阶段一：输入接收 (Inputs)
    ------------------------------------------------------------------------------------------------
    1. input_ids (文字身份证): [2, Seq_Len]
       Sample A: [101, "看", "!", 34, 34...34(196个), "和", 34, 34...34(196个)]
                 (注意：此时图片位置全是数字 34，这是毫无意义的占位符)
    
    2. pixel_values (原始像素): [2, 2, 3, 224, 224]
       包含了 4 张高清图片的 RGB 矩阵：[[猫, 狗], [苹果, 香蕉]]
    
    ------------------------------------------------------------------------------------------------
    阶段二：文本初加工 (Text Embedding)
    ------------------------------------------------------------------------------------------------
    代码: hidden_states = embed_tokens(input_ids)
    
    变化: ID 变成 向量。
    形状: [2, Seq_Len, 512] (假设 hidden_size=512)
    状态: 此时的 hidden_states 里：
          - "看", "!" -> 对应着有意义的文字向量。
          - "34" (占位符) -> 对应着初始化的随机向量/固定向量 (假肢)，完全没有视觉信息。
    
    ------------------------------------------------------------------------------------------------
    阶段三：视觉特征提取 (Vision Extraction)
    ------------------------------------------------------------------------------------------------
    目的: 把 pixel_values 里的像素变成向量，准备做器官移植。
    
    1. 拆解与编码 (Loop & CLIP):
       - 轮次 1: 提取所有样本的"第1张图" -> [猫, 苹果] -> 送入 CLIP -> 得到特征 Tensor_1
       - 轮次 2: 提取所有样本的"第2张图" -> [狗, 香蕉] -> 送入 CLIP -> 得到特征 Tensor_2
    
    2. 堆叠 (Stack):
       - vision_tensors = stack([Tensor_1, Tensor_2])
       - 形状: [2, 2, 196, 768] (Batch=2, Num=2, Patch=196, Clip_Dim=768)
       - 包含: [[猫特征, 狗特征], [苹果特征, 香蕉特征]]
    
    3. 降维投影 (Projection):
       - CLIP输出是768维，LLM需要512维。
       - vision_proj = Linear(vision_tensors) -> 形状变为 [2, 2, 196, 512]
    
    ------------------------------------------------------------------------------------------------
    阶段四：手术注入 (Surgery / count_vision_proj)
    ------------------------------------------------------------------------------------------------
    目的: 把 hidden_states 里的"假肢"(34对应的向量) 挖掉，换成"真肢"(vision_proj)。
    
    操作 (以 Sample A 为例):
       1. 扫描: 发现索引 [3:199] 是第一个占位区，[201:397] 是第二个占位区。
       2. 切除: 扔掉 hidden_states 中这些位置的向量。
       3. 植入: 
          - 在 [3:199] 填入 "猫特征" (196个向量)
          - 在 [201:397] 填入 "狗特征" (196个向量)
    
    状态: 此时 hidden_states 变成了真正的"多模态向量"。
          它包含: [文本向量, 猫的视觉向量, 文本向量, 狗的视觉向量]
    
    ------------------------------------------------------------------------------------------------
    阶段五：大脑思考 (Transformer Backbone)
    ------------------------------------------------------------------------------------------------
    代码: for layer in self.model.layers...
    
    过程: 
       这个混合向量进入 32 层 Transformer。
       Self-Attention 机制开始工作：
       - 文字 "看" 会注意到后面的 "猫特征"。
       - "狗特征" 会结合前面的 "和" 字。
       - 视觉和文字信息在这一步深度交融。
    
    ------------------------------------------------------------------------------------------------
    阶段六：输出预测 (Output Head)
    ------------------------------------------------------------------------------------------------
    代码: logits = lm_head(hidden_states)
    
    1. 归一化 (Norm): 整理数据分布。
    2. 映射 (Linear): 把 512维 向量映射回 32000维 (词表大小)。
    3. Logits: 
       模型会计算出下一个词的概率。
       比如看到 "看 [猫特征]"，模型可能会高概率预测下一个字是 "在"。
    
    最终返回: CausalLMOutputWithPast (包含 Logits, Loss, 以及为了加速下一次推理的 KV Cache)
    ================================================================================================

"""