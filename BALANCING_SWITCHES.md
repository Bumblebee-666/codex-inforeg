# Balancing Switches

This project now exposes two paper-guided, plug-in balancing slots.

## Keep

- `--use_cross_gate`
  - Keep as the main fusion-side innovation.
  - It only changes fusion features now, not the unimodal heads.

## Replace

- `--sample_weighting_method`
  - `none`: disable unimodal reweighting
  - `legacy_focal`: keep the older focal-style branch weighting
  - `confidence_gap`: weak-modality reweighting based on unimodal confidence gap

- `--balance_regularizer_method`
  - `none`: disable branch regularization
  - `legacy_cmob`: keep the older CMoB-style counterfactual regularizer
  - `head_gap`: regularize only the dominant unimodal head when the confidence gap is large

## Paper Mapping

- `confidence_gap`
  - Motivation: stronger modalities dominate multimodal optimization, so the weaker modality should receive extra optimization weight.
  - Related papers:
    - Peng et al., "Balanced Multimodal Learning via On-the-Fly Gradient Modulation," CVPR 2022  
      https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.html
    - Li et al., "Boosting Multi-modal Model Performance with Adaptive Gradient Modulation," ICCV 2023  
      https://openaccess.thecvf.com/content/ICCV2023/html/Li_Boosting_Multi-modal_Model_Performance_with_Adaptive_Gradient_Modulation_ICCV_2023_paper.html
    - Wang et al., "Asymmetric Reweighting Learning for Multimodal Fusion," ICCV 2025  
      https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Asymmetric_Reweighting_Learning_for_Multimodal_Fusion_ICCV_2025_paper.html

- `head_gap`
  - Motivation: decouple balancing pressure from the full backbone and apply it to the dominant head first, reducing interference with shared multimodal learning.
  - Related papers:
    - Wei et al., "Decoupled Gradient Learning for Multimodal Fusion," ICCV 2025  
      https://openaccess.thecvf.com/content/ICCV2025/html/Wei_Decoupled_Gradient_Learning_for_Multimodal_Fusion_ICCV_2025_paper.html
    - Li et al., "Boosting Multi-modal Model Performance with Adaptive Gradient Modulation," ICCV 2023  
      https://openaccess.thecvf.com/content/ICCV2023/html/Li_Boosting_Multi-modal_Model_Performance_with_Adaptive_Gradient_Modulation_ICCV_2023_paper.html

## Recommended Starting Point

- `--use_cross_gate`
- `--sample_weighting_method confidence_gap`
- `--balance_regularizer_method head_gap`
- Keep `--balance_scope heads` (head-gap uses heads by design)
- Keep old flags only for ablation or backward compatibility
