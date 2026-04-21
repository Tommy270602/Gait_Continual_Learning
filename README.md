# Gait_Continual_Learning

What have been so far that is worth mentioning:

PAPER REPLICATION
- Data Analysis of the dataset
- Replication of the paper in terms of Standard Model and comparison with CDML Model
- Perform the MIA Attack on different models to see the privacy/accuracy tradeoff
  
NOVEL ADDITIONS
- Created the knowledge distillation model, adapting the RCAT loss, and comparison with the CDML
- Combining the CDML and the Knowledge Distillation model to get an hybrid model
- Analyzed other methods that might improve privacy, but they didn't really work out
- explored advanced methods based on frequencies of the data
- Perform different attacks, both privacy related and performance degrading related, in particular IIA, Feature Space Probe and Backdoor attacks are explicative
- Perform some defenses that deal with adding noise in the output of the model to defend against attacks that rely on the model outputs
- Ablation studies on CMDL that deeply investigate how this approach differs from the standard method
- Analyzed and investigate a Transformer + CMDL architecture and compared it to CNN + CMDL