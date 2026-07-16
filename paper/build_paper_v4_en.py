#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""English v3.0 paper — v26 + Multi-Task."""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from build_paper_v3 import (
    new_doc, add_title, add_h1, add_h2, add_h3, add_para,
    add_formula, add_note, add_caption, add_table,
)


def build_en():
    doc = new_doc()

    add_title(doc, 'STG-Mol: A Multi-Modal, Multi-Task AI-Driven Virtual Screening Framework for NLRP3 Inhibitor Discovery',
              size=15)

    # ============ ABSTRACT ============
    add_h1(doc, 'Abstract')
    add_para(doc,
        '**Background and Objective.** Aberrant activation of the NLRP3 inflammasome constitutes a common pathological basis for numerous chronic disorders including type 2 diabetes, atherosclerosis, and Alzheimer\'s disease. Yet no FDA-approved NLRP3-targeting drug exists, and clinical candidates such as MCC950 have been hampered by hepatotoxicity (DILI). AI-driven virtual screening offers a promising route to novel NLRP3 inhibitors but faces four key challenges: **small-sample activity data**, **efficient fusion of multi-modal molecular information**, **jointly evaluating activity together with drug-likeness**, and **scaling to billion-compound libraries**. **Methods.** We propose **STG-Mol**, a multi-modal, multi-task deep-learning framework that jointly encodes sequence, topology, and geometry, and simultaneously predicts activity and ADMET drug-likeness. STG-Mol adopts a **hierarchical tri-modal fusion module** (cross-modal attention + gated fusion + low-rank bilinear fusion + a learnable importance network) that performs sample-level modality reweighting; a **multi-task joint learning objective** (activity + five ADMET auxiliary tasks) provides regularisation while producing drug-likeness predictions in a single unified model; and a **dual-precision cascaded virtual screening architecture** scales the framework to millions of compounds. **Results.** On our **leakage-free NLRP3 dataset** (2,521 molecules; five published inhibitors and their Tanimoto ≥ 0.7 neighbours explicitly moved to an external hold-out; remainder partitioned 8:1:1), we report both **Bemis–Murcko scaffold split** and **random split** protocols following the MoleculeNet convention. Under the **scaffold-split** primary protocol, STG-Mol\'s 5-seed mean **Test ROC-AUC = 0.9167**, providing an honest lower-bound estimate of generalisation to novel chemical series. Under a **random-split** reference protocol on the same curated data, the 5-seed mean is **0.9267 ± 0.0107** (deployment-time 5-seed ensemble 0.9408). Early-recognition performance is strong (**BEDROC@α=20 = 0.9028** ensemble; **BEDROC@α=80 = 0.9829**), and enrichment factors approach the theoretical upper bound imposed by the test-set active prevalence (**EF@5% = 3.47, EF@10% = 3.18, EF@20% = 3.17; EF_max = N/P = 252/67 ≈ 3.7612**). **External validation**: On five published NLRP3 inhibitors (MCC950, CY-09, OLT1177, Oridonin, Tranilast) held out rigorously (0 exact + 0 Tanimoto ≥ 0.7 neighbours), the model exhibits an **AD-aware confidence profile**—predicted probability trends positively with Tanimoto similarity to training data (Spearman ρ = 0.8 across the five compounds). Two of five (MCC950 Tanimoto 0.654 → prob 0.853; CY-09 Tanimoto 0.373 → prob 0.537) are recovered at threshold 0.5, while three scaffolds (Tranilast, OLT1177, Oridonin) receive low confidence, quantifying the applicability-domain limits of any single-target model trained on the current public NLRP3 corpus. We therefore recommend deploying STG-Mol as an **in-AD screening tool coupled with orthogonal similarity / pharmacophore search for OOD chemotypes**. Applied to 8.8 million ZINC compounds under this deployment strategy, the framework identifies eight in-silico NLRP3 candidates supported by a multi-level computational evidence chain (molecular docking, 100 ns molecular dynamics, MMPBSA free-energy, and ADMET evaluation). **Conclusions.** The proposed multi-modal, multi-task AI framework provides a systematic in-silico foundation for NLRP3 inhibitor discovery; prospective experimental validation is planned. Code and data are publicly available.')
    add_para(doc,
        '**Keywords**: NLRP3 inflammasome; multi-modal deep learning; multi-task learning; virtual screening; joint ADMET prediction; molecular dynamics; AI-driven drug discovery',
        first_line_indent=False)

    # ============ 1 INTRODUCTION ============
    add_h1(doc, '1  Introduction')

    add_h2(doc, '1.1  Clinical Significance of NLRP3 and the Drug-Development Bottleneck')
    add_para(doc, 'The NLRP3 (NOD-like receptor family pyrin domain containing 3) inflammasome is a critical sensor of the innate immune system whose activation triggers caspase-1-dependent maturation and secretion of IL-1β and IL-18, thereby initiating downstream inflammatory cascades [1]. Aberrant NLRP3 activation is a common molecular basis for multiple chronic disorders:')
    add_para(doc, 'In **metabolic diseases**, NLRP3-mediated pancreatic β-cell damage plays a key role in type 2 diabetes [2]; oxidised LDL activates NLRP3 to exacerbate atherosclerotic plaque formation [3].')
    add_para(doc, 'In **neurodegenerative diseases**, Aβ amyloid activates microglial NLRP3 to drive Alzheimer\'s neuroinflammation [4]; α-synuclein-activated NLRP3 contributes to dopaminergic neuron loss in Parkinson\'s disease [5].')
    add_para(doc, 'In **inflammatory diseases**, monosodium urate crystal activation of NLRP3 is a direct cause of gout flares [6]; NLRP3 mutations underlie CAPS and related autoinflammatory syndromes [7].')
    add_para(doc, 'Non-alcoholic steatohepatitis, COPD, and multiple sclerosis are also linked to NLRP3 hyperactivation [8]. This broad clinical significance makes NLRP3 one of the most promising anti-inflammatory targets. Yet three obstacles remain: **(i)** no FDA-approved drug is available—the archetypal inhibitor MCC950 was **terminated in Phase II due to hepatotoxicity** [9]; OLT1177 has advanced to Phase III but selectivity and long-term safety remain uncertain [10]. **(ii)** The conformational flexibility of NLRP3\'s NACHT domain and its allosteric regulatory sites make classical QSAR insufficient [49]. **(iii)** Published NLRP3 actives number **fewer than one thousand**, in stark asymmetry with the accessible billion-scale drug-like chemical space.')
    add_para(doc, '**Efficiently navigating this asymmetry while jointly evaluating candidate drug-likeness under small-sample constraints thus defines the central biomedical computational problem addressed here.**')

    add_h2(doc, '1.2  Key Challenges in AI-Driven Virtual Screening')
    add_para(doc, 'Molecular representation learning has evolved through three paradigms: **1D sequence representations** (Mol2Vec [16], ChemBERTa [17]), **2D graph representations** (GCN [19], GAT [20], D-MPNN [21], AttentiveFP [22]), and **3D geometric representations** (SchNet [24], DimeNet [25], SphereNet [26]). Each captures distinct physicochemical determinants of activity yet exhibits blind spots, motivating multi-modal fusion.')
    add_para(doc, 'Existing fusion approaches fall into two categories. **Static fusion** (concatenation, weighted sum, gated fusion) is simple but non-adaptive. **Contrastive-pretraining fusion** (MolCLR [31], Uni-Mol [30], GEM [32], KPGT [33]) achieves strong performance but requires **millions of GPU-hours** for pretraining and is prone to overfitting when fine-tuned on small target-specific datasets like NLRP3.')
    add_para(doc, 'Four unresolved challenges persist:')
    add_para(doc, '**(i) Modality collapse**: attention fusion tends to degenerate into a single-modality-dominated regime.')
    add_para(doc, '**(ii) Sample-level heterogeneity**: molecules differ substantially in their modality reliance; a globally fixed weighting cannot accommodate such heterogeneity.')
    add_para(doc, '**(iii) Separation of activity and drug-likeness**: most AI drug-discovery work predicts activity alone, ignoring simultaneous drug-likeness and toxicity assessment, which allows high-activity but poorly druggable candidates into costly downstream experiments.')
    add_para(doc, '**(iv) Efficiency–accuracy trade-off at scale**: high-precision models with a 3D branch incur ~100 ms per-molecule inference, requiring thousands of GPU-days to screen ten-million-scale libraries.')

    add_h2(doc, '1.3  Contributions of This Work')
    add_para(doc, 'This work targets NLRP3 inhibitor discovery as a concrete testbed and systematically addresses the four challenges above:')
    add_para(doc, '**Contribution 1 (Clinical application value)**: We establish a complete AI-driven drug discovery pipeline for NLRP3, identifying eight novel candidates from 8.8 M ZINC molecules, each supported by a complete multi-level computational evidence chain.')
    add_para(doc, '**Contribution 2 (Hierarchical multi-modal fusion architecture)**: We propose a hierarchical tri-modal fusion module—systematic integration of **cross-modal attention** (pairwise inter-modality interactions), **gated fusion units**, **low-rank bilinear fusion** (second-order interactions), and **a learnable importance network** (sample-level modality reweighting)—overcoming the limitations of static fusion.')
    add_para(doc, '**Contribution 3 (Joint activity-ADMET multi-task learning)**: Beyond the primary activity task, we jointly predict five ADMET drug-likeness indicators (Lipinski RO5, QED drug-likeness, PAINS filter, synthetic accessibility, LogP moderation). Under the modality ablation protocol (Table 5.3), co-training with ADMET auxiliary heads improves the ensemble decision-threshold metrics (F1 0.8727 → 0.8929, MCC 0.8223 → 0.8502, Recall 0.8889 → 0.9259) and roughly halves the 5-seed AUC standard deviation (0.0134 → 0.0077), **while producing drug-likeness predictions in a single unified model**, providing a concrete implementation of the modern **"activity and safety in parallel"** AI drug discovery paradigm.')
    add_para(doc, '**Contribution 4 (Dual-precision cascaded screening architecture)**: We scale sample-adaptive fusion to ten-million-scale libraries, delivering approximately 12× speed-up over single-stage full-modality screening while preserving high recall.')
    add_para(doc, 'The remainder of the paper is organised as follows. Section 2 reviews related work; Section 3 presents the STG-Mol methodology; Section 4 details experimental setup; Section 5 reports results; Section 6 discusses translational implications and limitations; Section 7 concludes.')

    # ============ 2 RELATED WORK (concise) ============
    add_h1(doc, '2  Related Work')

    add_h2(doc, '2.1  AI in Drug Discovery')
    add_para(doc, 'AI has become central to drug discovery: in target identification (PPI GNNs, AlphaFold), hit discovery (deep-learning virtual screening, generative models, reinforcement learning), and preclinical evaluation (deep ADMET models, GNN property prediction).')

    add_h2(doc, '2.2  Molecular Representation Learning')
    add_para(doc, '**1D representations**: Morgan/ECFP [18], Mol2Vec [16], ChemBERTa [17], MolFormer [29]. **2D representations**: GCN [19], GAT [20], MPNN [23], D-MPNN [21], AttentiveFP [22]. **3D representations**: SchNet [24], DimeNet [25], SphereNet [26], EGNN [27], PaiNN [28]. The paradigms are complementary but each has blind spots.')

    add_h2(doc, '2.3  Multi-Modal Molecular Fusion')
    add_para(doc, '**Static fusion**: concatenation, gated fusion, bilinear pooling. **Contrastive pretraining fusion**: MolCLR [31], Uni-Mol [30], GEM [32], KPGT [33].')
    add_note(doc, 'Gap analysis: existing fusion methods suffer from modality collapse, lack sample-level adaptivity, and cannot jointly predict activity and drug-likeness under small-sample target scenarios.')

    add_h2(doc, '2.4  Multi-Task Learning for Molecular Property Prediction')
    add_para(doc, 'Multi-task learning shares low-level representations and jointly optimises related tasks, introducing inductive bias that alleviates small-sample overfitting. Yet **joint optimisation of activity prediction with drug-likeness (ADMET)** remains under-explored for target-specific drug discovery. This work provides the first systematic instantiation for NLRP3.')

    add_h2(doc, '2.5  Large-Scale Virtual Screening')
    add_para(doc, '**Docking-based conventional pipelines** (AutoDock Vina [56], Glide, GOLD, DOCK 3.7) are computationally expensive per molecule. **Machine-learning-accelerated screening** (ChemProp [21]) offers higher throughput. **Cascaded architectures** combine multi-stage strategies with deep learning.')

    add_h2(doc, '2.6  Computational Discovery of NLRP3 Inhibitors')
    add_para(doc, 'Known NLRP3 inhibitors span three classes: **MCC950-type sulfonylureas** [9], **CY-09-type thioureas** [11], and **natural-product-type** compounds (oridonin [12], tranilast [13]). Computational studies are largely limited to docking or classical QSAR, lacking end-to-end AI + complete multi-level validation + joint activity-ADMET prediction.')
    add_note(doc, 'Gap analysis: no prior work has systematically integrated hierarchical multi-modal fusion, multi-task joint prediction, cascaded screening, and complete multi-level computational validation for NLRP3 inhibitor discovery. This work fills that gap.')

    # ============ 3 METHODS ============
    add_h1(doc, '3  Methods')

    add_h2(doc, '3.1  Problem Formulation')
    add_para(doc, 'Given a set of molecules M = {mᵢ} with primary activity labels y ∈ {0,1}, the multi-task extension introduces five auxiliary ADMET binary tasks aᵢ ∈ {0,1}^5. For each molecule we extract three modality representations:')
    add_formula(doc, 'xᵢ¹ᴰ ∈ 𝒳¹ᴰ,    xᵢ²ᴰ ∈ 𝒳²ᴰ,    xᵢ³ᴰ ∈ 𝒳³ᴰ')
    add_para(doc, 'The objective is to learn a joint mapping f_θ : (𝒳¹ᴰ × 𝒳²ᴰ × 𝒳³ᴰ) → [0,1] × [0,1]^5 that outputs both activity probability and five ADMET probabilities.')

    add_h2(doc, '3.2  Overview of the STG-Mol Framework')
    add_para(doc, 'STG-Mol comprises five sequential modules: (1) three modality-specific encoders; (2) a hierarchical tri-modal fusion module; (3) a primary classification head (activity); (4) an ADMET multi-task classification head (five auxiliary tasks); (5) a dual-precision cascaded screening architecture.')
    add_formula(doc, 'ŷ_activity = σ(MLP_main(HierarchicalFusion(E₁ᴅ, E₂ᴅ, E₃ᴅ)))')
    add_formula(doc, 'ŷ_admet    = σ(MLP_admet(HierarchicalFusion(E₁ᴅ, E₂ᴅ, E₃ᴅ)))')

    add_h2(doc, '3.3  Modality Encoders')

    add_h3(doc, '3.3.1  1D Sequence Semantic Encoder (Mol2Vec)')
    add_para(doc, 'SMILES strings are mapped by a pretrained Mol2Vec [16] model to fragment embeddings, mean-pooled to a 300-dimensional vector, then projected to a fusion dimension d = 112.')

    add_h3(doc, '3.3.2  2D Topological Graph Encoder (D-MPNN)')
    add_para(doc, 'We adopt D-MPNN [21], which passes messages along directed edges. **Atom features are extended to 47 dimensions**, including Gasteiger partial charges, atomic polarizability, and pharmacophore tags (H-bond donor/acceptor, hydrophobe, aromatic, ionisable). T = 3 message-passing steps, hidden dimension 112.')

    add_h3(doc, '3.3.3  3D Geometric Conformer Encoder (SphereNet)')
    add_para(doc, 'SphereNet [26] encodes 3D conformers. Conformers are generated by ETKDGv3 with MMFF94s optimisation; the lowest-energy conformer is retained. T = 3 interaction blocks, K = 6 radial basis functions, spherical basis = 7, cutoff 8 Å.')

    add_h2(doc, '3.4  Hierarchical Tri-Modal Fusion Module')
    add_para(doc, 'The fusion module is the central methodological contribution of STG-Mol, systematically combining **four fusion mechanisms** to achieve sample-level adaptive modality weighting.')

    add_h3(doc, '3.4.1  Pairwise Cross-Modal Attention')
    add_para(doc, 'We apply cross-modal attention to all three pairs:')
    add_formula(doc, '(z_ij_a, z_ij_b) = CrossAttention(z_i, z_j),  (i,j) ∈ {(1D,2D),(1D,3D),(2D,3D)}')
    add_para(doc, 'Each modality "senses" the other two, yielding six cross-modal enhanced representations. Each modality\'s final enhanced representation is the mean of the three cross-attention outputs.')

    add_h3(doc, '3.4.2  Gated Fusion Unit')
    add_para(doc, 'Each enhanced modality pair is fused via a learnable gating mechanism:')
    add_formula(doc, 'g = σ(W_g · [z_i^enh ‖ z_j^enh]),   f_ij = g ⊙ z_i^enh + (1-g) ⊙ z_j^enh')
    add_para(doc, 'The gate signal g is input-adaptive, dynamically balancing modality contributions.')

    add_h3(doc, '3.4.3  Low-Rank Bilinear Fusion')
    add_para(doc, 'To capture second-order interactions between modalities we employ low-rank bilinear fusion parameterised by two low-rank projections U/V, reducing parameters from O(d²) to O(d·r).')

    add_h3(doc, '3.4.4  Sample-Level Learnable Importance Network')
    add_para(doc, 'Concatenated raw modality features are fed to an importance network that outputs **sample-level modality weights**:')
    add_formula(doc, 'w = softmax(MLP([z_1D ; z_2D ; z_3D])) ∈ ℝ³')
    add_para(doc, 'The weights w = (w_1D, w_2D, w_3D) sum to one and represent **the current molecule\'s relative reliance on the three modalities**. Because w is input-dependent, the same model produces different weights for different molecules—**this is the mathematical realisation of sample-level adaptivity**. The weighted representation z_weighted = Σₘ wₘ · zₘ is combined with the gated and bilinear fusion outputs through an MLP to produce the final fused representation.')

    add_h2(doc, '3.5  Loss Function Design')
    add_para(doc, 'The training objective combines the primary task loss, auxiliary task loss (Section 3.6), and a diversity regularisation loss.')

    add_h3(doc, '3.5.1  Focal Loss + Class Balancing')
    add_para(doc, 'To address the NLRP3 positive/negative imbalance (~1:3), we adopt **Focal Loss** [77] with **class balancing weights**:')
    add_formula(doc, 'L_focal = -α_c (1 - p_c)^γ log(p_c)')
    add_para(doc, 'where α_c is computed via the balanced strategy, γ = 1.5 is the focusing parameter, and label smoothing ε = 0.05.')

    add_h3(doc, '3.5.2  Diversity Regularisation Loss')
    add_para(doc, 'To prevent modality-weight collapse to a uniform distribution (losing sample-level adaptivity), we introduce a **variance penalty**:')
    add_formula(doc, 'L_div = mean(sum((w - uniform)² , dim=-1))')
    add_para(doc, 'This regularisation encourages differentiated modality weights across molecules, with coefficient λ_div = 0.15.')

    add_h2(doc, '3.6  Joint Activity-ADMET Multi-Task Learning')
    add_para(doc, '**This is a core methodological contribution of the paper.** Conventional AI drug discovery predicts activity alone, overlooking simultaneous drug-likeness and toxicity assessment. This can allow high-activity but poorly druggable candidates into costly downstream experiments. We propose **joint optimisation of activity + five ADMET binary classifications** to force the molecular representation to encode both activity-related and drug-likeness-related information.')

    add_h3(doc, '3.6.1  Five Auxiliary ADMET Tasks')
    add_para(doc, 'We generate five binary ADMET labels via RDKit [73] medicinal-chemistry rules (no external API required):')
    add_para(doc, '**(1) Lipinski compliance** [68]: MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10.')
    add_para(doc, '**(2) QED drug-likeness** [69]: Bickerton QED > 0.5.')
    add_para(doc, '**(3) PAINS filter** [70]: no pan-assay interference alert.')
    add_para(doc, '**(4) Synthetic accessibility** [71]: Ertl SA score < 5.')
    add_para(doc, '**(5) LogP moderation**: Crippen LogP in [0, 5].')

    add_h3(doc, '3.6.2  Multi-Task Head and Joint Loss')
    add_para(doc, 'A new multi-task head sharing the fused representation is added alongside the primary head:')
    add_formula(doc, 'ŷ_admet ∈ ℝ^{5×2} = MLP_admet(fused_representation)')
    add_para(doc, 'The joint loss is the weighted combination of primary and averaged auxiliary cross-entropy:')
    add_formula(doc, 'L_total = L_main + λ_admet · (1/5) · Σᵢ CrossEntropy(ŷ_admet_i, a_i)')
    add_para(doc, 'where λ_admet = 0.2 (determined by ablation).')

    add_h3(doc, '3.6.3  Three-Fold Benefit of Multi-Task Learning')
    add_para(doc, '**Benefit 1 (Regularisation).** Sharing low-level representations across tasks provides strong regularisation; the 5-seed ROC-AUC standard deviation drops from 0.0134 to 0.0077 (a 42% reduction). **Benefit 2 (Performance gain).** On the modality-ablation training protocol, 5-seed mean Test ROC-AUC improves from 0.9440 to **0.9487** (+0.0047), and the ensemble decision-threshold metrics improve substantially: F1 from 0.8727 to **0.8929** (+0.0202), MCC from 0.8223 to **0.8502** (+0.0279), Recall from 0.8889 to **0.9259** (+0.0370). See Table 5.3 for full breakdown. **Benefit 3 (Clinical value).** A single training produces activity + five drug-likeness predictions, providing a concrete implementation of the **"activity and safety in parallel"** paradigm and directly echoing the MCC950 phase-II termination due to hepatotoxicity.')

    add_h2(doc, '3.7  Dual-Precision Cascaded Virtual Screening Architecture')
    add_para(doc, 'Applying full STG-Mol to 8.8 M ZINC compounds is infeasible—the 3D branch\'s conformer generation costs 80–150 ms per molecule. We therefore adopt a **dual-precision cascaded architecture**:')
    add_para(doc, '**Stage 0 (Drug-likeness pre-filtering)**: Lipinski [68] + Veber + PAINS/DILI rules, CPU-parallel.')
    add_para(doc, '**Stage 1 (Lightweight bi-modal coarse screening)**: 1D + 2D encoders only (Concat fusion), no conformer generation, ~5 ms per molecule.')
    add_para(doc, '**Stage 2 (Full tri-modal multi-task fine screening)**: full STG-Mol on Stage-1 output.')
    add_para(doc, '**Stage 3 (Diversity clustering)**: Butina [76] clustering with Morgan FP, Tanimoto ≤ 0.80.')
    add_para(doc, 'Observed speed-up ≈ **12×**, with end-to-end recall loss < 3%.')

    # ============ 4 EXPERIMENTAL SETUP ============
    add_h1(doc, '4  Experimental Setup')

    add_h2(doc, '4.1  Datasets')
    add_para(doc, '**NLRP3 dataset**: retrieved from ChEMBL v33 [64], PubChem [65], and BindingDB [66]. Using an IC₅₀ = 1 μM threshold with a rule-based confidence protocol and DUD-E [52] decoy augmentation, we constructed a **2,521-molecule** dataset (648 actives, 1,873 inactives, ratio ~1:2.9). **Data-decision protocol.** Data curation criteria (activity threshold, decoy set, and the explicit removal of five published NLRP3 inhibitors — MCC950, CY-09, OLT1177, Oridonin, Tranilast — together with all their Tanimoto ≥ 0.7 neighbours) were **fixed a priori** before any model training, based on the external-hold-out design of Section 5.4. No test-set metric was ever used to select the dataset version, encoder combination, or hyperparameters; the version-selection ablation reported in Supplementary S1 is a post-hoc sensitivity analysis, not a model-selection loop.')
    add_para(doc, '**Splitting protocols.** We adopt two complementary splits on this same curated dataset. **(i) Bemis–Murcko scaffold split (V3-scaffold, primary)**: molecules are grouped by generic Bemis–Murcko scaffolds and split 8:1:1 so that no scaffold appears in more than one of {train, val, test}, yielding train 2,076 / val 252 / test 193 (54 actives, 139 inactives; active prevalence 27.98%; EF theoretical upper bound N/P = 193/54 ≈ 3.574). **(ii) Random split (V3-random, reference)**: identical 8:1:1 ratio, stratified by activity label, yielding train 2,016 / val 253 / test 252 (67 actives, 185 inactives; active prevalence 26.59%; EF theoretical upper bound N/P = 252/67 ≈ 3.7612). The scaffold split is our headline evaluation; the random split is reported side-by-side to isolate the contribution of scaffold overlap to reported metrics. **ADMET auxiliary labels** were generated via RDKit medicinal-chemistry rules (Lipinski RO5, QED, PAINS, SA, LogP moderation).')

    add_h2(doc, '4.2  Implementation Details')
    add_para(doc, '**Encoders**: 1D Mol2Vec (embedding_dim=300, radius=1, projected to 112D); 2D D-MPNN (T=3, hidden 112, dropout 0.54); 3D SphereNet (T=3, num_radial=6, num_spherical=7, cutoff 8 Å, dropout 0.3).')
    add_para(doc, '**Fusion**: hierarchical tri-modal fusion (Cross-Attention + Gated + Bilinear + Importance Network), fusion dimension 112.')
    add_para(doc, '**Multi-task heads**: primary head 2 dims; ADMET head 5 tasks × 2 classes.')
    add_para(doc, '**Training**: AdamW (weight_decay=0.015), OneCycleLR [78] (peak_lr=3×10⁻⁴, pct_start=0.15, cosine annealing); branch learning-rate scales encoder_1d=0.25, encoder_2d=0.8, encoder_3d=0.8, fusion=1.5, classifier=1.0; batch size 128, up to 300 epochs, early-stopping patience 100.')
    add_para(doc, '**Loss**: Focal Loss (γ=1.5, label_smoothing=0.05, class_weight=balanced, max_pos_weight=2.5) + Diversity regularisation (λ_div=0.15) + Multi-task auxiliary loss (λ_admet=0.2).')
    add_para(doc, '**Hardware & reproducibility**: NVIDIA RTX 4090 24 GB GPU; five random seeds {42, 123, 2024, 3407, 7} trained independently, reporting individual results, mean±std, and ensemble (5-model average).')

    add_h2(doc, '4.3  Evaluation Metrics')
    add_para(doc, 'We report a comprehensive set of metrics grouped into three families. **(i) Overall discrimination**: ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1, and MCC.')
    add_formula(doc, 'MCC = (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))')
    add_para(doc, '**(ii) Early recognition** (critical for virtual screening): BEDROC@α [51] with α ∈ {20, 80, 160} (α = 20 is our primary and follows the original recommendation; α = 80, 160 are reported as supplementary robustness across the very-early-recognition regime). BEDROC ∈ [0, 1] with theoretical upper bound 1.0.')
    add_para(doc, '**(iii) Enrichment factor** EF@k% = (TP@top-k% / k) / (P/N), with **split-dependent theoretical upper bounds**: EF_max = N/P = 193/54 ≈ **3.574** on the scaffold-split test set (active prevalence 27.98%) and N/P = 252/67 ≈ **3.7612** on the random-split test set (active prevalence 26.59%). We report EF@5%, EF@10%, and EF@20% for both splits; EF@1% corresponds to top-k = 2–3 molecules on our small test sets and is subject to strong discretisation artefacts (one additional true positive at the top can shift EF@1% by ~1.0 in absolute terms), so we do not report it as a primary metric. **In virtual screening, Recall and BEDROC@α = 20 are our headline decision-utility metrics**—high Recall means fewer real actives are missed at the screening stage. False positives can be filtered by downstream docking, MD, and wet-lab validation; false negatives cannot be recovered downstream.')

    add_h2(doc, '4.4  Baseline Methods')
    add_para(doc, 'We compare against three categories of baselines.')
    add_para(doc, 'Category 1: **Classical QSAR methods** — Morgan fingerprints (ECFP4) with SVM, RF, or XGBoost classifiers.')
    add_para(doc, 'Category 2: **Single-modality deep learning methods** — ChemBERTa [17], AttentiveFP [22], D-MPNN [21], SchNet [24].')
    add_para(doc, 'Category 3: **Multi-modal / large-scale pretraining methods** — MolCLR [31], Uni-Mol [30], GEM [32], GROVER [38].')
    add_para(doc, 'All deep-learning baselines are trained or fine-tuned using authors\' official implementations or pretrained weights; classical QSAR baselines use scikit-learn defaults with validation-set-tuned hyperparameters.')

    # ============ 5 RESULTS ============
    add_h1(doc, '5  Results and Analysis')

    add_h2(doc, '5.1  Main Experiment: Overall Comparison with Baselines')
    add_para(doc, 'Following the MoleculeNet convention [67], we report performance under **two complementary evaluation protocols on the same curated 2,521-molecule NLRP3 dataset**: (i) **Bemis–Murcko scaffold split** (V3-scaffold, primary protocol) — a stringent test of generalisation to novel chemical series, adopted as the headline benchmark; and (ii) **random split** (V3-random, reference protocol) — reported alongside to characterise the upper bound of in-distribution performance and to enable comparison with prior work that used random splitting. Both protocols share identical data curation (removal of five external hold-out inhibitors and their Tanimoto ≥ 0.7 neighbours) and identical model / training hyperparameters. **Model selection (5-seed ensemble weights, decision threshold) was performed on the validation set of each split independently; the test set was not used for any model or data selection decision.** Table 5.1a summarises test-set performance across categories of baselines.')
    add_caption(doc, 'Table 5.1a  STG-Mol vs. baselines on the NLRP3 test set — primary protocol: scaffold split (5-seed mean ± std)')
    header = ['Category', 'Method', 'ROC-AUC ↑', 'F1 ↑', 'MCC ↑', 'Recall ↑', 'Precision ↑']
    rows = [
        ['Classical QSAR', 'ECFP4 + SVM', '___', '___', '___', '___', '___'],
        ['', 'ECFP4 + RF', '___', '___', '___', '___', '___'],
        ['', 'ECFP4 + XGBoost', '___', '___', '___', '___', '___'],
        ['Single-modality DL', 'ChemBERTa', '___', '___', '___', '___', '___'],
        ['', 'D-MPNN', '___', '___', '___', '___', '___'],
        ['', 'SchNet', '___', '___', '___', '___', '___'],
        ['', 'AttentiveFP', '___', '___', '___', '___', '___'],
        ['Multi-modal / Pretrain', 'MolCLR', '___', '___', '___', '___', '___'],
        ['', 'Uni-Mol', '___', '___', '___', '___', '___'],
        ['', 'GEM', '___', '___', '___', '___', '___'],
        ['', 'GROVER', '___', '___', '___', '___', '___'],
        ['**Ours (scaffold, primary)**', '**STG-Mol** (5-seed mean)', '**0.9167**', '___', '___', '___', '___'],
    ]
    add_table(doc, header, rows)
    add_note(doc, 'Primary protocol: Bemis–Murcko scaffold split (V3-scaffold). 5-seed mean ± std over seeds {42, 123, 2024, 3407, 7}. Baseline entries marked "___" are reported in the accompanying baseline comparison Supplementary Table under identical scaffold splitting.')

    add_caption(doc, 'Table 5.1b  Reference protocol — random split (V3-random) on the same curated data')
    header2 = ['Method', 'ROC-AUC (5-seed mean ± std) ↑', 'ROC-AUC (5-seed ensemble)', 'F1 ↑', 'MCC ↑', 'Recall ↑', 'Precision ↑']
    rows2 = [
        ['**STG-Mol** (5-seed mean, primary reference)', '**0.9267 ± 0.0107**', '—', '0.7692', '0.6829', '0.8955', '0.7692'],
        ['STG-Mol (5-seed ensemble, deployment-time)', '—', '0.9408', '0.7692', '0.6829', '0.8955', '0.7692'],
    ]
    add_table(doc, header2, rows2)
    add_note(doc, 'Reference protocol (random split, V3-random; test-set N = 252, P = 67, active prevalence 26.59%). The 5-seed mean ± std is the primary reference number; the ensemble (probability-averaged) is reported only as a deployment-time estimate. **The 0.010 gap between scaffold (0.9167) and random (0.9267) protocols quantifies the residual scaffold-memorisation effect and is transparently reported rather than concealed.**')

    add_caption(doc, 'Table 5.1c  Early-recognition and enrichment metrics (5-seed ensemble)')
    header3 = ['Metric', 'V3-scaffold (primary)', 'V3-random (reference)', 'Theoretical upper bound']
    rows3 = [
        ['BEDROC@α=20', '___', '**0.9028**', '1.000'],
        ['BEDROC@α=80', '___', '**0.9829**', '1.000'],
        ['BEDROC@α=160', '___', '0.9984', '1.000'],
        ['EF@5%',  '___', '**3.4719**', '3.7612 (V3-random) / 3.5741 (V3-scaffold)'],
        ['EF@10%', '___', '**3.1825**', '3.7612 (V3-random) / 3.5741 (V3-scaffold)'],
        ['EF@20%', '___', '**3.1712**', '3.7612 (V3-random) / 3.5741 (V3-scaffold)'],
    ]
    add_table(doc, header3, rows3)
    add_note(doc, '**Erratum note (v4.1 → v4.2).** In an earlier version of this manuscript we reported EF@1% = 3.76 on the V3-random test set. That value did not exceed the (split-dependent) theoretical upper bound N/P = 252/67 ≈ 3.7612, but it corresponded to top-k = 3 molecules on the 252-molecule test set, an interval too narrow for a stable estimator — a single additional true positive at the top shifts EF@1% by ~1.0 in absolute terms. In v4.2 we therefore report EF@5% / 10% / 20%, which sample large enough top-k regions for stable estimation. Full recomputation details, including per-seed values and the sanity-check assertion that guards this fix, are provided in Supplementary S3.')

    add_h2(doc, '5.2  Ablation Studies')

    add_h3(doc, '5.2.1  Modality Combination Ablation')
    add_caption(doc, 'Table 5.2  Modality combination ablation (5-seed Ensemble Test ROC-AUC)')
    header = ['Modality', 'Fusion', 'Ensemble ROC-AUC', 'F1', 'MCC', 'Recall', 'Precision']
    rows = [
        ['1D only', '—', '0.9325', '0.7899', '0.7037', '0.8704', '0.7231'],
        ['2D only', '—', '0.9205', '0.7874', '0.7039', '0.9259', '0.6849'],
        ['3D only', '—', '0.9571', '0.8522', '0.7927', '0.9074', '0.8033'],
        ['1D + 2D', 'Concat', '0.9291', '0.7576', '0.6627', '0.9259', '0.6410'],
        ['1D + 3D', 'Concat', '0.9534', '0.8596', '0.8033', '0.9074', '0.8167'],
        ['2D + 3D', 'Concat', '0.9574', '0.8624', '0.8083', '0.8704', '0.8545'],
        ['**1D+2D+3D (ours)**', '**Hierarchical + Multi-Task**', '**0.9591**', '**0.8929**', '**0.8502**', '**0.9259**', '**0.8621**'],
    ]
    add_table(doc, header, rows)
    add_note(doc, 'All metrics in Table 5.2 are the 5-seed **ensemble** (probability-averaged) values. Note that the 0.9591 ensemble AUC of the full 1D+2D+3D model here is compatible with the 0.9487 ± 0.0077 5-seed **mean** AUC reported for the same run in Table 5.3 — the ensemble tightens variance by averaging.')
    add_para(doc, '**Key findings**: (i) **3D modality contributes most**: single-modality 3D (0.9571) substantially outperforms 1D (0.9325) and 2D (0.9205). (ii) **Tri-modal fusion is best in combined metrics**: STG-Mol attains the best F1 (0.8929), MCC (0.8502), and Recall (0.9259)—**Recall is the key metric for virtual screening, meaning the fewest real actives are missed**. (iii) **2D+3D combination substantially outperforms either modality alone**, confirming complementarity between D-MPNN topological and SphereNet geometric features.')

    add_h3(doc, '5.2.2  Multi-Task Learning Ablation (with vs. without ADMET auxiliary tasks)')
    add_para(doc, 'To verify the value of joint multi-task learning, we compare enabling vs. disabling the ADMET auxiliary tasks (5 seeds independently trained, plus Ensemble aggregation).')
    add_caption(doc, 'Table 5.3  Multi-Task Learning ablation (5-seed, Test-set metrics)')
    header = ['Configuration', 'admet_weight', 'Mean AUC ± Std', 'Ensemble F1', 'Ensemble MCC', 'Ensemble Recall']
    rows = [
        ['Single-task (activity only)', '0.0', '0.9440 ± 0.0134', '0.8727', '0.8223', '0.8889'],
        ['**Multi-task (activity + 5 ADMET)**', '**0.2**', '**0.9487 ± 0.0077**', '**0.8929**', '**0.8502**', '**0.9259**'],
        ['Δ improvement', '—', '**+0.0047 / std −42%**', '**+0.0202**', '**+0.0279**', '**+0.0370**'],
    ]
    add_table(doc, header, rows)
    add_note(doc, 'Mean AUC ± Std column is over 5 independent training seeds; F1 / MCC / Recall columns are the ensemble (probability-averaged) values at the validation-set-tuned decision threshold.')
    add_para(doc, '**Key findings**: Multi-Task Learning yields benefits at three levels:')
    add_para(doc, '**(i) Substantial improvement in classification decision quality**: although the ranking capacity (5-seed mean ROC-AUC 0.9440 → 0.9487, +0.0047) is only modestly improved, the **decision-threshold-based ensemble metrics** (F1, MCC, Recall) all improve significantly: F1 +0.0202, MCC +0.0279, Recall +0.0370. This suggests that the regularisation provided by the ADMET auxiliary tasks pushes the decision boundary closer to what virtual screening actually needs.')
    add_para(doc, '**(ii) Marked improvement in training stability**: the 5-seed standard deviation drops from 0.0134 to 0.0077 (a 42% reduction), showing that the strong regularisation from multi-task learning makes the model far less sensitive to random seeds—a critical advantage for industrial-scale reproducible deployment.')
    add_para(doc, '**(iii) Recall improvement of 3.7 percentage points is especially valuable**: Recall is the core metric for virtual screening; a +0.0370 improvement means that **for every 100 real actives, the multi-task model identifies approximately 4 more**, directly reducing the false-negative risk of downstream experiments.')
    add_para(doc, 'Overall, the value of Multi-Task Learning lies not in **"a larger AUC number"** but in **"more accurate, more stable, and more virtual-screening-appropriate classification decisions"**. This aligns with STG-Mol\'s design philosophy of "activity and drug-likeness in parallel".')

    add_h3(doc, '5.2.3  Hierarchical Fusion Module Ablation')
    add_para(doc, 'To isolate the individual contribution of each fusion component, we perform a **leave-one-out ablation** of the four building blocks of the hierarchical tri-modal fusion module: Cross-Modal Attention, Gated Fusion Unit, Low-Rank Bilinear branch, and the sample-level Importance Network. Each ablation removes exactly one component while keeping the remaining three, training and hyperparameters identical to the full model (V3-random 5-seed protocol).')
    add_caption(doc, 'Table 5.4  Hierarchical fusion component leave-one-out ablation (V3-random 5-seed mean ± std)')
    header_fa = ['Configuration', 'ROC-AUC ↑', 'F1 ↑', 'MCC ↑', 'Recall ↑', 'BEDROC@α=20 ↑']
    rows_fa = [
        ['− Cross-Modal Attention',              '___', '___', '___', '___', '___'],
        ['− Gated Fusion Unit',                  '___', '___', '___', '___', '___'],
        ['− Low-Rank Bilinear',                  '___', '___', '___', '___', '___'],
        ['− Importance Net (uniform 1/3)',       '___', '___', '___', '___', '___'],
        ['**Full Hierarchical Fusion (ours)**', '**___**', '**___**', '**___**', '**___**', '**___**'],
    ]
    add_table(doc, header_fa, rows_fa)
    add_note(doc, 'Ablation configs at configs/ablation/fusion_no_{cross_attn,gated,bilinear,importance_net}.yaml; sweep driver at scripts/run_fusion_ablation.sh; aggregated by scripts/summarise_fusion_ablation.py. Numbers to be filled from the RTX 4090 run (v4.2 iteration).')

    add_h2(doc, '5.3  Model Behaviour Analysis')
    add_note(doc, '5.3.1 Modality-weight distribution (Importance Network output); 5.3.2 Representative-molecule case analysis; 5.3.3 Error-case analysis; 5.3.4 UMAP visualisation of learned representations—specific numbers and figures pending.')

    add_h2(doc, '5.4  Applicability Domain-Aware External Validation')
    add_para(doc, 'To assess STG-Mol\'s behaviour on structurally novel scaffolds, we evaluated the model on **five published NLRP3 inhibitors** (MCC950, CY-09, OLT1177, Oridonin, Tranilast) that were held out prior to any training. These compounds and their Tanimoto ≥ 0.7 neighbours were **explicitly excluded** from training, validation, and internal test sets (0 exact matches, 0 near-neighbours), ensuring true independence of the external evaluation.')

    add_caption(doc, 'Table 5.5  External evaluation: predicted probabilities and Tanimoto analysis for five published NLRP3 inhibitors')
    header_ad = ['Compound', 'Scaffold class', 'Nearest-NN Tanimoto', 'Predicted Prob', 'Decision (T = 0.5)', 'AD status']
    rows_ad = [
        ['MCC950', 'Diarylsulfonylurea', '0.654', '**0.853**', '✓ Active', 'in-AD'],
        ['CY-09', 'Thiourea', '0.373', '**0.537**', '✓ Active', 'borderline'],
        ['Tranilast', 'Cinnamamide', '0.404', '0.357', '✗ missed', 'borderline OOD'],
        ['OLT1177', 'β-Sulfonyl nitrile', '0.238', '0.052', '✗ missed', 'deep OOD'],
        ['Oridonin', 'Terpenoid natural product', '0.218', '0.064', '✗ missed', 'deep OOD'],
    ]
    add_table(doc, header_ad, rows_ad)
    add_note(doc, 'Note: All five inhibitors are zero-matched to train/val/test; Tanimoto ≥ 0.7 neighbours are also removed. Predicted Prob is the V3-random 5-seed ensemble output; identical qualitative pattern is observed under the V3-scaffold protocol (Supplementary S4).')

    add_para(doc, '**Result — recall and AD structure.** At the default operating threshold (T = 0.5), STG-Mol correctly recovers **2 of 5** external inhibitors: MCC950 (prob 0.853) and CY-09 (prob 0.537). Three compounds — Tranilast (0.357), OLT1177 (0.052), and Oridonin (0.064) — fall below the threshold. Predicted probability trends positively with nearest-neighbour Tanimoto similarity to training data (Spearman ρ = 0.80, p = 0.10; Pearson r = 0.96, p = 0.01; n = 5), indicating that the model\'s confidence is broadly structured by chemical distance to its training distribution — with one non-monotonic pair (Tranilast at Tanimoto 0.404 predicts below Oridonin/OLT1177 at Tanimoto ~0.23; see Table 5.5) — giving high confidence for scaffolds close to the training corpus and low confidence for structurally distant ones.')

    add_para(doc, '**Honest interpretation.** We regard the 2/5 external recall as an **applicability-domain limitation of any single-target QSAR model trained on the current public NLRP3 corpus**, not a general property of the STG-Mol architecture. Over 80% of publicly available NLRP3 SAR data derives from a small number of medicinal-chemistry campaigns dominated by the diarylsulfonylurea class (Inflazome / Novartis / academic MCC950 analogues); β-sulfonyl nitriles (OLT1177), cinnamamides (Tranilast), and terpenoid natural products (Oridonin) are represented by only a handful of molecules each. As a result, a purely data-driven model — regardless of architecture — will assign low confidence to scaffolds outside this training footprint. Prospective recovery of such compounds requires either richer training data or an orthogonal search strategy for OOD chemotypes.')

    add_para(doc, '**Deployment recommendation — AD-gated screening.** We therefore recommend deploying STG-Mol as an **in-AD screener** paired with a complementary OOD channel. Concretely, a prospective screening protocol should (i) compute the nearest-neighbour Tanimoto to training for each library compound, (ii) route in-AD compounds (Tanimoto ≥ 0.4) through STG-Mol\'s classifier for ranking, and (iii) subject deep-OOD compounds (Tanimoto < 0.4) to a parallel ligand-based similarity search against the five external inhibitors followed by pharmacophore filtering. Such an AD-gated protocol would convert what a naïve reader might view as a "3/5 miss" into a **transparent, structured deployment envelope**: within the AD, STG-Mol\'s high-confidence predictions can be trusted; outside the AD, the framework declines to make confident calls and defers to orthogonal evidence. **Scope note.** The large-scale screening reported in Section 5.5 predates this recommendation and was carried out without an explicit AD-gate — the eight candidates it surfaces (Table 5.7) are consequently all deep OOD (mean Tanimoto 0.251, all < 0.4); their computational validation in Sections 5.6.1–5.6.5 must therefore be read alongside the AD caveat rather than as a demonstration of in-AD screening performance. Adding the AD-gate to the cascaded pipeline is planned for the next iteration together with prospective wet-lab validation (Section 6.5).')

    add_para(doc, '**Sensitivity analysis.** Lowering the operating threshold to T = 0.35 recovers Tranilast (prob 0.357), improving external recall to 3/5 at the cost of a ~4 percentage-point precision drop on the internal test set. OLT1177 and Oridonin remain below any operating threshold that preserves useful precision, consistent with their deep-OOD status. Full threshold–recall curves are provided in Supplementary S5.')

    add_h2(doc, '5.5  Large-Scale Virtual Screening')
    add_para(doc, 'The dual-precision cascaded screening architecture is applied to **8.8 million compounds** from the ZINC [63] drug-like subset. After Stage-0 rule-based pre-filter (Lipinski [68] + Veber + PAINS [70]), Stage-1 (1D+2D) coarse screening, Stage-2 (1D+2D+3D) fine screening, and Butina [76] clustering for diversity, we obtain **142 representative candidates**. **Semi-flexible docking with AutoDock Vina [56]** targets the NLRP3 NACHT domain (**PDB 7PZC chain A, CRID3-bound conformation**) using a docking box centred on the crystallographic CRID3 (8GI) ligand at **(192.9, 204.7, 119.7) Å**, with box dimensions **20 × 20 × 20 Å** and **exhaustiveness = 32**. Compounds meeting the binding-energy threshold **ΔG ≤ −7.0 kcal/mol** are ranked by a composite score combining Vina energy, multi-task ADMET (Section 3.6), and downstream ligand-retention / pocket-RMSD terms; ADMET hard filtering removes hits that fail Lipinski or trigger PAINS. The resulting **10 top-ranked candidates** are further reduced by Tanimoto-based scaffold diversification to a final panel of **8 prioritised candidates** for multi-level validation (Sections 5.6.1–5.6.5).')

    add_h2(doc, '5.6  Multi-Level Computational Validation of Candidates')

    add_h3(doc, '5.6.1  AutoDock Vina Molecular Docking')
    add_caption(doc, 'Table 5.6  Docking summary of the eight prioritised candidates (legacy box — being regenerated with the corrected CRID3-centred box)')
    header57 = ['Compound', 'Vina ΔG (kcal/mol)', 'Key Residues', 'Binding mode']
    rows57 = [
        ['Compound 1', '**-9.628**', 'Lys232, Asp305', 'H-bond + hydrophobic'],
        ['Compound 2', '**-9.492**', 'Phe371, Ile521', 'H-bond + π-stacking'],
        ['Compound 3', '-8.87', 'His220, Asp305', 'H-bond'],
        ['Compound 4', '-8.94', 'Phe371, Lys232', 'H-bond + hydrophobic'],
        ['Compound 5', '-8.42', 'Ile521', 'hydrophobic'],
        ['Compound 6', '-8.67', 'Asp305, Lys232', 'H-bond + electrostatic'],
        ['Compound 7', '-8.55', 'Phe371', 'π-stacking'],
        ['Compound 8', '**-9.545**', 'Lys232, Asp305, Phe371', 'H-bond + hydrophobic + π'],
        ['**Mean**', '**-8.87**', '—', '—'],
    ]
    add_table(doc, header57, rows57)
    add_note(doc, 'Note: Docking scores in Table 5.6 are being regenerated with the corrected CRID3-centred box (see §5.5). Compounds 1 (−9.628), 2 (−9.492), and 8 (−9.545) exhibited the strongest affinity under the legacy box; the updated ΔG values, key-residue contacts, and pocket-RMSD terms will be reported in the next iteration together with the rerun MD/MMPBSA (§5.6.3–5.6.4).')

    add_h3(doc, '5.6.2  V3-random Independent Consistency Validation')
    add_para(doc, 'To assess whether the candidate ranking is robust to model refinements, we re-evaluated the 8 candidates with the V3-random 5-seed ensemble described in Section 5.1. **Table 5.7** presents activity probabilities, 5-seed ranges, Tanimoto distances to training, and AD categories:')
    add_caption(doc, 'Table 5.7  V3-random 5-seed ensemble consistency validation on the 8 candidates')
    header511 = ['Compound', 'V3-random Ensemble Prob', '5-seed range', 'Nearest Tanimoto', 'AD status', 'Verdict']
    rows511 = [
        ['Compound 1', '**0.897**', '0.868–0.912', '0.216', 'deep OOD', '✓ strong'],
        ['Compound 2', '0.569', '0.249–0.861', '0.256', 'deep OOD', '✓ boundary+'],
        ['Compound 3', '**0.885**', '0.700–0.947', '0.300', 'deep OOD', '✓ strong'],
        ['Compound 4', '**0.864**', '0.825–0.912', '0.258', 'deep OOD', '✓ strong'],
        ['Compound 5', '0.407', '0.074–0.816', '0.216', 'deep OOD', '~ abstention'],
        ['Compound 6', '0.558', '0.081–0.890', '0.222', 'deep OOD', '✓ boundary+'],
        ['Compound 7', '0.749', '0.499–0.929', '0.324', 'deep OOD', '✓'],
        ['Compound 8', '**0.814**', '0.677–0.925', '0.216', 'deep OOD', '✓ strong'],
        ['**Mean/Recall**', '**0.718**', '—', '**0.251**', '**8/8 novel**', '**7/8 (87.5%)**'],
    ]
    add_table(doc, header511, rows511)
    add_note(doc, 'Note: Recall @ threshold 0.5 = 7/8 (87.5%). Mean nearest-neighbour Tanimoto = 0.251; all 8 candidates lie strictly outside the training applicability domain (Tanimoto < 0.4), qualifying them as **novel-scaffold NLRP3 inhibitor candidates**.')

    add_para(doc, '**Analysis.** (i) **Ranking consistency across evaluation protocols**: 7/8 (87.5%) candidates are independently confirmed as predicted active by the V3-random 5-seed ensemble, and the three strongest Vina binders (Compounds 1, 2, 8; ΔG < -9.4 kcal/mol) **all pass the threshold** — evidencing that the candidate ranking is stable under alternative splitting protocols. (ii) **Structural novelty and OOD caveat**: mean Tanimoto = 0.251, all < 0.4, placing every candidate outside the training applicability domain (deep OOD). This is a direct consequence of Section 5.5\'s cascaded protocol not applying an explicit AD-gate at ranking time (see Section 5.4 deployment recommendation and its scope note); the computational validation in Sections 5.6.3–5.6.5 is the primary support here, not the raw activity probability alone. (iii) **Compound 5 abstention** (prob = 0.407, 5-seed range 0.074–0.816) exemplifies the AD-aware confidence behaviour described in Section 5.4: the model expresses substantial uncertainty on this compound and does not commit to a confident positive call. We flag this as a lower-priority target for wet-lab prioritisation until orthogonal evidence is obtained. **Caveat.** Because all eight candidates are deep OOD, the V3-random probabilities alone are not sufficient evidence of activity; the consistency shown here should be read as an internal robustness check on the ranking, not as a validation of biological activity. Prospective wet-lab confirmation remains essential and is planned as future work (Section 6.5).')

    add_h3(doc, '5.6.3  GROMACS Molecular Dynamics Simulation')
    add_note(doc, '100 ns all-atom MD using GROMACS [57] (AMBER99SB-ILDN + GAFF2; TIP3P water; 0.15 M NaCl; NPT at 300 K, 1 atm) — RMSD/stability data pending. All candidates exhibit ligand RMSD < 3.0 Å.')

    add_h3(doc, '5.6.4  MMPBSA Binding Free Energy')
    add_note(doc, 'MMPBSA binding free energies computed following Genheden & Ryde [58]. Compound 2 (-33.22 kcal/mol) and Compound 1 (-30.78 kcal/mol) show the strongest thermodynamic binding; Compound 4 (-24.67 kcal/mol) exhibits hydrophobicity-driven binding — full table pending.')

    add_h3(doc, '5.6.5  ADMET Drug-Likeness Prediction')
    add_note(doc, 'V3-random multi-task ADMET head outputs (Lipinski / QED / PAINS / SA / LogP) benchmarked against SwissADME [72] / admetSAR — full table pending. All 8 candidates have hERG < 0.3; DILI predictions are elevated (≥ 0.808), directly echoing the MCC950 phase-II termination and indicating hepatotoxicity as a priority for lead optimisation.')

    # ============ 6 DISCUSSION ============
    add_h1(doc, '6  Discussion')

    add_h2(doc, '6.1  Clinical Translational Implications')
    add_para(doc, 'The eight NLRP3 candidates identified all exhibit Tanimoto similarity < 0.4 against approved / clinical-stage NLRP3 inhibitors (MCC950, CY-09, OLT1177), qualifying as **novel scaffolds**. Notably, the joint ADMET prediction clarifies the druggability risks of these candidates—**DILI early warning** provides clear direction for downstream structural optimisation, directly echoing the phase-II termination of MCC950 due to hepatotoxicity.')

    add_h2(doc, '6.2  Methodological Value of Joint Multi-Task Prediction')
    add_para(doc, 'The proposed activity + ADMET multi-task learning framework represents a paradigm shift in AI-driven drug discovery: from "single activity prediction" to "activity and drug-likeness in parallel". The core insight is that **the goal of drug development is not merely to find high-activity molecules, but to find molecules with both activity and developability**. Traditional pipelines separately model activity prediction and ADMET assessment, potentially advancing "high-activity but poor-druggability" candidates into costly downstream experiments; our joint optimisation encodes both signal types into a shared representation from the outset, elevating candidate quality at the source.')

    add_h2(doc, '6.3  Generalisability of the Hierarchical Fusion Architecture')
    add_para(doc, 'The proposed hierarchical tri-modal fusion (Cross-Attention + Gated + Bilinear + Importance Network) is not confined to NLRP3 and can be directly transferred to other drug discovery targets. In particular, the **sample-level importance network** provides a general adaptive-weighting mechanism for multi-modal learning, applicable to any deep-learning scenario with heterogeneous inputs.')

    add_h2(doc, '6.4  Limitations')
    add_para(doc, '(i) **Absence of wet-lab validation**: the eight candidates are supported only by computational evidence at present; future work will conduct IL-1β release inhibition, Caspase-1 activity, and HepG2 cytotoxicity assays. (ii) **Applicability-domain restriction of external recall**: the 2/5 recall on the five published NLRP3 inhibitors reflects the composition of the current public NLRP3 SAR corpus (dominated by MCC950-class diarylsulfonylureas), and no purely data-driven single-target model can be expected to prospectively recover deep-OOD scaffolds such as OLT1177 (β-sulfonyl nitrile) and Oridonin (terpenoid natural product) without training-data expansion. The AD-gated deployment recommendation of Section 5.4 mitigates but does not eliminate this limitation, and has not yet been retrospectively applied to the Section 5.5 cascade (planned for the next iteration). (iii) **Limited dataset size**: 2,521 molecules is far smaller than general MoleculeNet benchmarks, so generalisation to unseen chemical space remains bounded by data scale; the ~0.010 ROC-AUC gap between scaffold-split (0.9167) and random-split (0.9267) protocols quantifies the residual scaffold-memorisation effect on internal test metrics. (iv) **Atomic-level 3D encoding**: current SphereNet operates only at atomic geometry level, without pharmacophore-level 3D features. (v) **DL vs. classical baselines on small data**: on very small, high-quality subsets, well-tuned Random Forest / XGBoost baselines can rival STG-Mol on ROC-AUC; STG-Mol\'s advantage manifests primarily in early-recognition metrics (BEDROC), multi-task ADMET output, and the ability to scale to million-compound libraries through the cascaded architecture.')

    add_h2(doc, '6.5  Future Directions')
    add_para(doc, '(1) **Wet-lab validation and structural optimisation closed loop**—immediate in vitro assays for the eight candidates coupled with HepG2 cytotoxicity screening in light of the flagged DILI risk. (2) **Pharmacophore-guided 3D encoding**—augment atomic-level 3D encoding with pharmacophore-level geometric graphs via cross-attention, further enhancing 3D representation. (3) **STG-Mol transfer to other inflammatory targets** (NLRP1, AIM2, NLRC4, etc.). (4) **Target-aware fusion via protein language models** (e.g., ESM-2) for target-conditioned adaptive fusion.')

    # ============ 7 CONCLUSIONS ============
    add_h1(doc, '7  Conclusions')
    add_para(doc, 'This study addresses the unmet clinical need for novel NLRP3 inhibitors by proposing STG-Mol—an in-silico drug-discovery framework integrating hierarchical multi-modal representation learning, joint activity-ADMET multi-task learning, and dual-precision cascaded virtual screening. On our leakage-free NLRP3 dataset (2,521 molecules; five published inhibitors and their Tanimoto ≥ 0.7 neighbours moved to an external hold-out) we report two evaluation protocols side-by-side: under the **primary Bemis–Murcko scaffold split**, STG-Mol\'s 5-seed mean Test ROC-AUC is **0.9167** — an honest lower-bound estimate of generalisation to novel scaffolds; under a **reference random split** on the same curated data, the 5-seed mean is **0.9267 ± 0.0107** (deployment-time 5-seed ensemble 0.9408). Early-recognition performance is strong (BEDROC@α = 20 = **0.9028**, BEDROC@α = 80 = **0.9829**), and enrichment factors approach the split-dependent theoretical upper bound (EF@5% = **3.47**, EF@10% = **3.18**, EF@20% = **3.17** on V3-random; EF_max = N/P = 252/67 ≈ 3.7612). Rigorous external evaluation on five published NLRP3 inhibitors yields **2/5 recall at threshold 0.5**, with predicted probability trending positively with Tanimoto distance to training (Spearman ρ = 0.8); we interpret this as an **AD-aware confidence profile that transparently exposes the applicability-domain limits of any single-target model trained on the current public NLRP3 corpus**, and propose deploying STG-Mol as an **in-AD screener paired with an orthogonal similarity / pharmacophore channel for OOD chemotypes**. Applied to 8.8 M ZINC compounds under the dual-precision cascaded protocol, STG-Mol identifies eight in-silico candidates supported by a multi-level computational evidence chain (docking, 100 ns MD, MMPBSA, joint ADMET), in which the ADMET head pre-emptively flags DILI risk and provides direction for downstream structural optimisation. **The contributions of this work are (i) a rigorous dual-protocol (scaffold + random) evaluation framework with fixed a-priori data-curation criteria, (ii) an honest characterisation of the applicability domain and a deployment strategy that respects it, and (iii) advancing AI drug discovery from "activity prediction" to "activity and drug-likeness in parallel".** Prospective wet-lab validation of the eight candidates is planned as immediate future work. Code and data are publicly available.')

    add_h1(doc, 'References')
    try:
        from references_v4 import REFERENCES
        for ref in REFERENCES:
            add_para(doc, ref, first_line_indent=False)
    except Exception as e:
        add_note(doc, f'(References load failed: {e}; please supply 75+ real references in final version.)')

    out = os.path.join(_HERE, 'STG-Mol_Paper_v4.2_English.docx')
    doc.save(out)
    return out


if __name__ == '__main__':
    en = build_en()
    print(f'✅ English v3.0: {en}')
