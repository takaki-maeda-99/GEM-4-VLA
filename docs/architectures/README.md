# Architecture: gemma_siglip_action_query

## Purpose

Use SigLIP for visual tokens, Gemma as the semantic backbone, and learnable action query tokens to predict continuous action chunks.

This architecture is intended for imitation learning with multi-view images, language instructions, optional proprioception, and continuous robot actions.

## High-level flow

```mermaid
flowchart TB                                                                                                                                                                                                                             
      subgraph BATCH[Batch from X-VLA-style Dataloader]                                                                                                                                                                                    
          DI[domain_id: B]                                                                                                                                                                                                                 
          SI[scene_img: B,3,224,224]                                                                                                                                                                                                       
          WI[wrist_img: B,3,224,224]                                                                                                                                                                                                       
          TX[language: B,L tokens]                                                                                                                                                                                                         
          PR[proprio: B,Dp]                                                                                                                                                                                                                
          LA[last_action_chunk: B,T,Da]                                                                                                                                                                                                    
          AT[target action: B,T,Da]                                                                                                                                                                                                        
      end                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                           
      subgraph VIS[SigLIP Vision Encoder shared]                                                                                                                                                                                           
          SI --> SCN[scene tokens: B,Ns,d]                                                                                                                                                                                                 
          WI --> WRN[wrist tokens: B,Nw,d]                                                                                                                                                                                                 
      end                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                           
      subgraph DA[Domain-Aware Projections from X-VLA]                                                                                                                                                                                     
          DI -. domain_id .-> SP_HUB[(SoftPrompt Hub<br/>nn.Embedding<br/>num_domains x Ks x H)]                   
          DI -. domain_id .-> SCN_PROJ[Scene Proj DomainAwareLinear]                                               
          DI -. domain_id .-> WRN_PROJ[Wrist Proj DomainAwareLinear]                                               
          DI -. domain_id .-> PROP_PROJ[Proprio Proj DomainAwareLinear]                                            
          DI -. domain_id .-> LACT_PROJ[LastActionChunk Proj DomainAwareLinear]                                    
          DI -. domain_id .-> AQ_HUB[(ActionQuery Hub<br/>nn.Embedding<br/>num_domains x T x H)]                   
      end                                                                                                          

      SP_HUB --> SP_TOK[soft prompt: B,Ks,H]                                              
      SCN --> SCN_PROJ --> SCN_TOK[scene tokens: B,Ns,H]  
      WRN --> WRN_PROJ --> WRN_TOK[wrist tokens: B,Nw,H]                                                           
      TX --> TXT_EMB[Gemma text embed: B,Lt,H]           
      AQ_HUB --> AQ_TOK[action queries: B,T,H]           
                                                         
      subgraph PACK[Sequence Packing]                                                                              
          SP_TOK --> SEQ                                 
          SCN_TOK --> SEQ   
          TXT_EMB --> SEQ                                
          WRN_TOK --> SEQ                                
          AQ_TOK --> SEQ[Sequence:<br/>SoftPrompt | Scene | Text | Wrist | ActionQuery<br/>B,Ks+Ns+Lt+Nw+T,H]        
      end                                                
                                                         
      SEQ --> GEMMA[Gemma 3n E2B<br/>output_hidden_states=True]                                                      
      GEMMA --> ALL_H[hidden states all layers<br/>L+1, B, S, H]                                                     
                            
      PR --> PROP_PROJ --> PROP_TOK[proprio token: B,1,H]                                                            
      LA --> LACT_PROJ --> LACT_TOK[last action seq: B,T,H]                                                          
                                                                                                                   
      subgraph HEAD[Action Head VLA-Adapter style]                                                                 
          ALL_H -. per layer i .-> H_T[task tokens h_t<br/>= scene+wrist+text patches]                             
          ALL_H -. per layer i .-> H_A[action tokens h_a<br/>= ActionQuery positions]                                
          LACT_TOK --> X0[init x = LastActionChunk + proprio fused<br/>B,T,H]                                        
          PROP_TOK --> X0                                 
          X0 --> BLK[24 x MLPResNetBlock<br/>each block does cross-attn:<br/>x ⟂ h_t i+1, h_a i+1, p<br/>residual + FFN]
          BLK --> OUT[Action Decoder<br/>DomainAwareLinear H to Da]                                                  
      end                                            
```