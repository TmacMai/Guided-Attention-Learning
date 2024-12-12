from torch import nn
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model
from transformers.models.bert.modeling_bert import BertPooler
import global_configs
from global_configs import DEVICE
from modules.transformer import TransformerEncoder
import torch
import numpy as np
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
import torch.nn.functional as F
import math
class DebertaModel(DebertaV2PreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = (global_configs.TEXT_DIM, global_configs.ACOUSTIC_DIM,
                                              global_configs.VISUAL_DIM)
        self.config = config
        self.pooler = BertPooler(config)
        model = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
        self.model = model.to(DEVICE)
        self.beta_shift = multimodal_config.beta_shift
        self.lambda2 = multimodal_config.lambda2
        self.loss = nn.MSELoss(reduction = 'none')
        self.ratio = multimodal_config.ratio
        self.cc_margin = multimodal_config.cc_margin
        self.cs_margin = multimodal_config.cs_margin
        self.d_l = multimodal_config.share_dim
        self.attn_dropout = multimodal_config.drop_prob
        self.unimodal_dropout = multimodal_config.dropout_unimodal
        self.unimodal_head = multimodal_config.transformer_head
        self.label_dim = 1
        self.proj_a = nn.Conv1d(ACOUSTIC_DIM, self.d_l, kernel_size=multimodal_config.kernel_size, stride=1, padding=(multimodal_config.kernel_size-1)//2, bias=False)
        self.proj_l = nn.Linear(TEXT_DIM, self.d_l, bias=False)
        self.proj_v = nn.Conv1d(VISUAL_DIM, self.d_l, kernel_size=multimodal_config.kernel_size, stride=1, padding=(multimodal_config.kernel_size-1)//2, bias=False)
        self.transa = self.get_network(self_type='a', layers=multimodal_config.transformer_layer)
        self.transv = self.get_network(self_type='v', layers=multimodal_config.transformer_layer)

        self.coff =  multimodal_config.loss_t_ratio

        self.predict_attention = nn.Sequential(
            nn.Linear(self.d_l, multimodal_config.inter_dim * 2),
            nn.Tanh(),   
            nn.Linear(multimodal_config.inter_dim * 2, self.d_l),
            nn.Tanh(),   
            nn.Linear(self.d_l, 1),
        )


        self.predictor = nn.Sequential(
            nn.Linear(self.d_l, multimodal_config.inter_dim),
            nn.ReLU(),  
            nn.Linear(multimodal_config.inter_dim, self.label_dim),
        )

        self.predictor_av = nn.Sequential(
            nn.Linear(self.d_l, multimodal_config.inter_dim),
            nn.ReLU(),  
            nn.Linear(multimodal_config.inter_dim, self.label_dim),
        )


        self.predictor_al = nn.Sequential(
            nn.Linear(self.d_l, multimodal_config.inter_dim),
            nn.ReLU(),  
            nn.Linear(multimodal_config.inter_dim, self.label_dim),
        )

        self.predictor_vl = nn.Sequential(
            nn.Linear(self.d_l, multimodal_config.inter_dim),
            nn.ReLU(),  
            nn.Linear(multimodal_config.inter_dim, self.label_dim),
        )

        self.iterations = 0
        self.pretrain_iterations = multimodal_config.pretrain_iterations

        self.attention_a = nn.Linear(self.d_l, self.d_l)
        self.attention_v = nn.Linear(self.d_l, self.d_l)
        self.attention_l = nn.Linear(self.d_l, self.d_l)
        self.attention_av = nn.Linear(self.d_l, self.d_l)
        self.attention_al = nn.Linear(self.d_l, self.d_l)
        self.attention_vl = nn.Linear(self.d_l, self.d_l)
        self.attention_av2 = nn.Linear(self.d_l, self.d_l)
        self.attention_al2 = nn.Linear(self.d_l, self.d_l)
        self.attention_vl2 = nn.Linear(self.d_l, self.d_l)

        self.LayerNorm = nn.LayerNorm(self.d_l)
        self.LayerNorm2 = nn.LayerNorm(self.d_l)
        self.LayerNorm3 = nn.LayerNorm(self.d_l)
  

        self.initalize = False
        self.feature = False
       # self.init_weights()

    def get_network(self, self_type='l', layers=3):
        if self_type in ['l', 'a', 'v']:
            embed_dim, attn_dropout, transformer_droput, transformer_head = self.d_l, self.attn_dropout, self.unimodal_dropout, self.unimodal_head
        elif self_type in ['lav']:
            embed_dim, attn_dropout, transformer_droput, transformer_head = self.d_l * 3, self.attn_dropout, self.unimodal_dropout, self.unimodal_head
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads= transformer_head, 
                                  layers=layers,
                                  attn_dropout=attn_dropout,
                                  relu_dropout=transformer_droput,
                                  res_dropout= transformer_droput,
                                  embed_dropout= transformer_droput,
                                  attn_mask= False)

    def forward(
            self,
            input_ids,
            visual,
            acoustic,
            label_ids = None,
            feature = False,
    ):
        if feature:
            self.feature  = True
        y = label_ids.view(-1,)

        ####unimodal networks
        embedding_output = self.model(input_ids)
        x = embedding_output[0]
        output_l = self.proj_l(x)

        acoustic = acoustic.transpose(1, 2)
        visual = visual.transpose(1, 2)

        acoustic = self.proj_a(acoustic)
        visual = self.proj_v(visual)

        acoustic = acoustic.permute(2, 0, 1)  ### t, bs, dim
        visual = visual.permute(2, 0, 1)

        output_a = self.transa(acoustic)
        output_v = self.transv(visual)

        output_a = output_a.permute(1, 0, 2)   #### bs, t, dim
        output_v = output_v.permute(1, 0, 2)

        #######cross-modal attention
        k_av, k_al, k_vl = torch.cat([output_a, output_v], dim = 1), torch.cat([output_a, output_l], dim = 1), torch.cat([output_v, output_l], dim = 1)

        attention_a, attention_v, attention_l = self.attention_a(output_a), self.attention_v(output_v), self.attention_l(output_l)
        attention_av, attention_vl, attention_al = self.attention_av(k_av), self.attention_vl(k_vl), self.attention_al(k_al)

        av_matrix_pre  = torch.bmm(attention_l, attention_av.transpose(2,1))/math.sqrt(k_av.shape[-1])
        al_matrix_pre  = torch.bmm(attention_v, attention_al.transpose(2,1))/math.sqrt(k_av.shape[-1])
        vl_matrix_pre  = torch.bmm(attention_a, attention_vl.transpose(2,1))/math.sqrt(k_av.shape[-1])


        av_matrix = F.softmax(av_matrix_pre , dim = -1)
        al_matrix = F.softmax(al_matrix_pre , dim = -1)
        vl_matrix = F.softmax(vl_matrix_pre , dim = -1)


        attention_output_vl = self.LayerNorm(torch.bmm(vl_matrix, k_vl)).mean(dim = 1)
        attention_output_al = self.LayerNorm2(torch.bmm(al_matrix, k_al)).mean(dim = 1)
        attention_output_av = self.LayerNorm3(torch.bmm(av_matrix, k_av)).mean(dim = 1)


        attend_vl  = self.predict_attention(attention_output_vl)
        attend_al  = self.predict_attention(attention_output_al)
        attend_av  = self.predict_attention(attention_output_av)

        attend_sum = torch.exp(attend_vl) + torch.exp(attend_al) + torch.exp(attend_av)
        weight_vl = torch.exp(attend_vl) / attend_sum 
        weight_al = torch.exp(attend_al) / attend_sum 
        weight_av = torch.exp(attend_av) / attend_sum 

        fusion = attend_vl * attention_output_vl + attend_al * attention_output_al + attend_av * attention_output_av

        pooled_output = self.predictor(fusion)


        if not self.training:
            noise = torch.rand(attention_a.shape[0], attention_a.shape[-1]).to(y.device)
            attend_noise  = self.predict_attention(noise)
        ### auxiliary loss
        loss_p, loss_i, loss_t = 0, 0, 0
        loss_o, loss_neutral, loss_r = 0, 0, 0
        if self.training:
            self.iterations += 1
            #### output attention ####
            predict_vl = self.predictor_vl(attention_output_vl.detach())
            predict_al = self.predictor_al(attention_output_al.detach())
            predict_av = self.predictor_av(attention_output_av.detach())

            loss_vl = self.loss(predict_vl.squeeze(), y).reshape(-1,)
            loss_al = self.loss(predict_al.squeeze(), y).reshape(-1,)
            loss_av = self.loss(predict_av.squeeze(), y).reshape(-1,)
            loss_neutral = (loss_vl + loss_al + loss_av).mean()

            if feature:
                index1, index2, index3 = np.array([i for i in range(y.shape[0])]), np.array([i for i in range(y.shape[0])]), np.array([i for i in range(y.shape[0])])
                filter_index = (loss_vl[index1] > loss_al[index1]).nonzero().view(-1).cpu().numpy()
                for i in range(len(y)):
                    if i not in list(filter_index):
                         loss_o += torch.relu(attend_al[i] + self.cs_margin - attend_vl[i]).mean()
                    else:
                         loss_o += torch.relu(-attend_al[i] + self.cs_margin + attend_vl[i]).mean()

                filter_index = (loss_vl[index2] > loss_av[index2]).nonzero().view(-1).cpu().numpy()
                for i in range(len(y)):
                    if i not in list(filter_index):
                         loss_o += torch.relu(attend_av[i] + self.cs_margin - attend_vl[i]).mean()
                    else:
                         loss_o += torch.relu(-attend_av[i] + self.cs_margin + attend_vl[i]).mean()


                filter_index = (loss_av[index3] > loss_al[index3]).nonzero().view(-1).cpu().numpy()
                for i in range(len(y)):
                    if i not in list(filter_index):
                         loss_o += torch.relu(attend_al[i] + self.cs_margin - attend_av[i]).mean()
                    else:
                         loss_o += torch.relu(-attend_al[i] + self.cs_margin + attend_av[i]).mean()

                loss_o = loss_o / len(y)

                noise = torch.rand(1, attention_a.shape[-1]).to(y.device)
                attend_noise  = self.predict_attention(noise)
                loss_o += torch.relu(attend_noise + self.cs_margin - attend_av).mean()
                loss_o += torch.relu(attend_noise + self.cs_margin - attend_al).mean()
                loss_o += torch.relu(attend_noise + self.cs_margin - attend_vl).mean()

            
            batch = [i for i in range(attention_a.shape[1])]
            index4 = np.random.choice(batch, int(attention_a.shape[1] * self.ratio), replace=False)
            index5 = np.random.choice(batch, int(attention_a.shape[1] * self.ratio), replace=False)
            index6 = np.random.choice(batch, int(attention_a.shape[1] * self.ratio), replace=False)

            noise = torch.rand(attention_a.shape[0], attention_a.shape[1], attention_a.shape[-1]).to(y.device)

            batch = np.array([1 for i in range(attention_a.shape[1])])
            batch[index4] = 0
            index_array = np.expand_dims(batch, axis = 0)
            index_array = np.expand_dims(index_array, axis = 2)
            index_array = torch.LongTensor(index_array).to(output_l.device)
            attenion_a_noise = output_a * index_array + noise * (1 - index_array)
            k_av_anoise = attenion_a_noise
            k_al_anoise = attenion_a_noise
            k_av_anoise = self.attention_av(k_av_anoise)
            k_al_anoise = self.attention_al(k_al_anoise)

            k_av_aenhance = output_a * self.lambda2 + output_l * (1 - self.lambda2)
            k_al_aenhance = output_a * self.lambda2 + output_v * (1 - self.lambda2)
            k_av_aenhance = self.attention_av(k_av_aenhance)
            k_al_aenhance = self.attention_al(k_al_aenhance)


            batch = np.array([1 for i in range(attention_v.shape[1])])
            batch[index5] = 0
            index_array = np.expand_dims(batch, axis = 0)
            index_array = np.expand_dims(index_array, axis = 2)
            index_array = torch.LongTensor(index_array).to(output_l.device)
            attenion_v_noise = output_v * index_array + noise * (1 - index_array)
            k_av_vnoise = attenion_v_noise
            k_vl_vnoise = attenion_v_noise
            k_av_vnoise = self.attention_av(k_av_vnoise)
            k_vl_vnoise = self.attention_vl(k_vl_vnoise)

            k_av_venhance = output_v * self.lambda2 + output_l * (1 - self.lambda2)
            k_vl_venhance = output_v * self.lambda2 + output_a * (1 - self.lambda2)
            k_av_venhance = self.attention_av(k_av_venhance)
            k_vl_venhance = self.attention_vl(k_vl_venhance)


            batch = np.array([1 for i in range(attention_l.shape[1])])
            batch[index6] = 0
            index_array = np.expand_dims(batch, axis = 0)
            index_array = np.expand_dims(index_array, axis = 2)
            index_array = torch.LongTensor(index_array).to(output_l.device)
            attenion_l_noise = output_l * index_array + noise * (1 - index_array)
            k_al_lnoise = attenion_l_noise
            k_vl_lnoise = attenion_l_noise
            k_al_lnoise = self.attention_al(k_al_lnoise)
            k_vl_lnoise = self.attention_vl(k_vl_lnoise)

            k_al_lenhance = output_l * self.lambda2 + output_v * (1 - self.lambda2)
            k_vl_lenhance = output_l * self.lambda2 + output_a * (1 - self.lambda2)
            k_al_lenhance = self.attention_al(k_al_lenhance)
            k_vl_lenhance = self.attention_vl(k_vl_lenhance)


            ###########################
            av_matrix_anoise_pre = torch.bmm(attention_l, k_av_anoise.transpose(2,1))/math.sqrt(k_av.shape[-1])
            av_matrix_vnoise_pre = torch.bmm(attention_l, k_av_vnoise.transpose(2,1))/math.sqrt(k_av.shape[-1])
            al_matrix_anoise_pre = torch.bmm(attention_v, k_al_anoise.transpose(2,1))/math.sqrt(k_av.shape[-1])
            al_matrix_lnoise_pre = torch.bmm(attention_v, k_al_lnoise.transpose(2,1))/math.sqrt(k_av.shape[-1])
            vl_matrix_vnoise_pre = torch.bmm(attention_a, k_vl_vnoise.transpose(2,1))/math.sqrt(k_av.shape[-1])
            vl_matrix_lnoise_pre = torch.bmm(attention_a, k_vl_lnoise.transpose(2,1))/math.sqrt(k_av.shape[-1])


            av_matrix_anoise = F.softmax(av_matrix_anoise_pre, dim = -1)
            av_matrix_vnoise = F.softmax(av_matrix_vnoise_pre, dim = -1)
            al_matrix_anoise = F.softmax(al_matrix_anoise_pre, dim = -1)
            al_matrix_lnoise = F.softmax(al_matrix_lnoise_pre, dim = -1)
            vl_matrix_vnoise = F.softmax(vl_matrix_vnoise_pre, dim = -1)
            vl_matrix_lnoise = F.softmax(vl_matrix_lnoise_pre, dim = -1)

            av_matrix1 = av_matrix_pre[:, :, :av_matrix_anoise.shape[-1]]
            al_matrix1 = al_matrix_pre[:, :, :al_matrix_anoise.shape[-1]]
            vl_matrix1 = vl_matrix_pre[:, :, :vl_matrix_vnoise.shape[-1]]
            av_matrix2 = av_matrix_pre[:, :, av_matrix_anoise.shape[-1]:]
            al_matrix2 = al_matrix_pre[:, :, al_matrix_anoise.shape[-1]:]
            vl_matrix2 = vl_matrix_pre[:, :, vl_matrix_vnoise.shape[-1]:]

            loss_i += torch.relu(av_matrix_anoise_pre[:, :, index4] + self.cc_margin - av_matrix1[:, :, index4]).mean()      
            loss_i += torch.relu(av_matrix_vnoise_pre[:, :, index5] + self.cc_margin - av_matrix2[:, :, index5]).mean()      
            loss_i += torch.relu(al_matrix_anoise_pre[:, :, index4] + self.cc_margin - al_matrix1[:, :, index4]).mean()      
            loss_i += torch.relu(al_matrix_lnoise_pre[:, :, index6] + self.cc_margin - al_matrix2[:, :, index6]).mean()    
            loss_i += torch.relu(vl_matrix_lnoise_pre[:, :, index6] + self.cc_margin - vl_matrix2[:, :, index6]).mean()      
            loss_i += torch.relu(vl_matrix_vnoise_pre[:, :, index5] + self.cc_margin - vl_matrix1[:, :, index5]).mean() 


            ##################################
            av_matrix_aehance = torch.bmm(attention_l, k_av_aenhance.transpose(2,1))/math.sqrt(k_av.shape[-1])
            av_matrix_vehance = torch.bmm(attention_l, k_av_venhance.transpose(2,1))/math.sqrt(k_av.shape[-1])
            al_matrix_aehance = torch.bmm(attention_v, k_al_aenhance.transpose(2,1))/math.sqrt(k_av.shape[-1])
            al_matrix_lehance = torch.bmm(attention_v, k_al_lenhance.transpose(2,1))/math.sqrt(k_av.shape[-1])
            vl_matrix_vehance = torch.bmm(attention_a, k_vl_venhance.transpose(2,1))/math.sqrt(k_av.shape[-1])
            vl_matrix_lehance = torch.bmm(attention_a, k_vl_lenhance.transpose(2,1))/math.sqrt(k_av.shape[-1])

            av_matrix1 = av_matrix_pre [:, :, :av_matrix_vehance.shape[-1]]
            al_matrix1 = al_matrix_pre [:, :, :al_matrix_lehance.shape[-1]]
            vl_matrix1 = vl_matrix_pre [:, :, :vl_matrix_vehance.shape[-1]]
            av_matrix2 = av_matrix_pre [:, :, av_matrix_vehance.shape[-1]:]
            al_matrix2 = al_matrix_pre [:, :, al_matrix_lehance.shape[-1]:]
            vl_matrix2 = vl_matrix_pre [:, :, vl_matrix_vehance.shape[-1]:]

            loss_r += torch.relu(-torch.diagonal(av_matrix_aehance, dim1=-2, dim2=-1) + self.cc_margin + torch.diagonal(av_matrix1, dim1=-2, dim2=-1)).mean()      
            loss_r += torch.relu(-torch.diagonal(av_matrix_vehance, dim1=-2, dim2=-1) + self.cc_margin + torch.diagonal(av_matrix2, dim1=-2, dim2=-1)).mean()      
            loss_r += torch.relu(-torch.diagonal(al_matrix_aehance, dim1=-2, dim2=-1) + self.cc_margin + torch.diagonal(al_matrix1, dim1=-2, dim2=-1)).mean()      
            loss_r += torch.relu(-torch.diagonal(al_matrix_lehance, dim1=-2, dim2=-1) + self.cc_margin + torch.diagonal(al_matrix2, dim1=-2, dim2=-1)).mean()    
            loss_r += torch.relu(-torch.diagonal(vl_matrix_lehance, dim1=-2, dim2=-1) + self.cc_margin + torch.diagonal(vl_matrix2, dim1=-2, dim2=-1)).mean()      
            loss_r += torch.relu(-torch.diagonal(vl_matrix_vehance, dim1=-2, dim2=-1) + self.cc_margin + torch.diagonal(vl_matrix1, dim1=-2, dim2=-1)).mean()     


        if self.training:
            return pooled_output, loss_o, loss_i,  loss_r, loss_neutral, (av_matrix_anoise, al_matrix_lnoise, vl_matrix_vnoise), (av_matrix, al_matrix, vl_matrix), (weight_av, weight_al, weight_vl)
         #   return pooled_output, loss_o, loss_i,  loss_r, loss_neutral, (av_matrix_aehance, al_matrix_aehance, vl_matrix_vehance), (av_matrix, al_matrix, vl_matrix), (weight_av, weight_al, weight_vl)
        else:
            return pooled_output, loss_o, loss_i,  loss_r, loss_neutral, (av_matrix, al_matrix, vl_matrix), (attend_av, attend_al, attend_vl, attend_noise), (weight_av, weight_al, weight_vl)


class DeBertaForSequenceClassification(DebertaV2PreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.dberta = DebertaModel(config, multimodal_config)
        self.init_weights()

    def forward(
            self,
            input_ids,
            visual,
            acoustic,
            label_ids = None,
            feature = False,
    ):
        pooled_output, p_loss, i_loss, loss_r, loss_unimodal, x1, x2, x3 = self.dberta(
            input_ids,
            visual,
            acoustic,
            label_ids,
            feature
        )

        return pooled_output, p_loss, i_loss, loss_r, loss_unimodal, x1, x2, x3
