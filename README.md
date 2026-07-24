# Privacy Attacks and Countermeasures in Biometric Gait Identification
Gait recognition from inertial measurement unit (IMU) signals has emerged as a
practical biometric modality for continuous and transparent user authentication.
When deployed in practice, such systems must enrol new users over time through
\emph{continual learning}, which introduces two intertwined challenges:
catastrophic forgetting and privacy leakage.

This thesis investigates these two problems jointly, using the Code Division
Modulation Layer (CDML) framework as a starting point. The thesis extends this
baseline in several directions and evaluates the resulting methods on three
publicly available IMU datasets: WU~Gait, UCI~HAR and WISDM.

On the learning side, a knowledge distillation strategy (CDML+KD), a
Lipschitz constraint approach (LiDER) and a wavelet-based generative replay method
(WGR-CDML) are proposed to reduce forgetting without storing raw biometric data.
On the architectural side, Low-Rank Adaptation (LoRA) adapters are introduced as a
parameter-efficient continual learning mechanism. A state-space model backbone
based on the Mamba architecture (GaitMamba) is also evaluated
as an alternative to the convolutional baseline.

Privacy is evaluated against four attack families: Membership Inference Attacks,
Identity Inference Attacks, Feature Space Inference and Backdoor Attacks.

Finally, a set of cross-dataset experiments is presented, exploring transfer
learning between datasets, adapter warm-starting and joint multi-dataset
classification, to assess the generalisability of the proposed framework beyond
the single-dataset setting.
  