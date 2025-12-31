# Baseline Methods Graph

This page visualizes baseline-method relationships extracted from meta files.

Each component represents a family of related methods, showing how newer papers build upon previous baseline methods.

## PagedAttention Family

*7 methods, 9 relationships*

```mermaid
flowchart LR
    classDef defaultNode fill:#4A90E2,stroke:#2E5C8A,stroke-width:2px,color:#fff
    classDef rootNode fill:#50C878,stroke:#2E7D4E,stroke-width:3px,color:#fff
    classDef leafNode fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px,color:#fff
    linkStyle default stroke:#9370DB,stroke-width:2px

    2026_FlashOverlap["FlashOverlap[2026]"]
    class 2026_FlashOverlap leafNode
    2025_TileLink["TileLink[2025]"]
    class 2025_TileLink defaultNode
    2023_PagedAttention["PagedAttention[2023]"]
    class 2023_PagedAttention rootNode
    2024_FLUX["FLUX[2024]"]
    class 2024_FLUX rootNode
    2024_Async_TP["Async-TP[2024]"]
    class 2024_Async_TP rootNode
    2025_NanoFlow["NanoFlow[2025]"]
    class 2025_NanoFlow defaultNode
    2025_TokenWeave["TokenWeave[2025]"]
    class 2025_TokenWeave leafNode

    2025_TileLink ==>|" "| 2025_TokenWeave
    linkStyle 0 stroke:#9370DB,stroke-width:2.5px
    2023_PagedAttention ==>|" "| 2025_TileLink
    linkStyle 1 stroke:#FF6347,stroke-width:2.5px
    2023_PagedAttention ==>|" "| 2025_NanoFlow
    linkStyle 2 stroke:#20B2AA,stroke-width:2.5px
    2023_PagedAttention ==>|" "| 2025_TokenWeave
    linkStyle 3 stroke:#FFD700,stroke-width:2.5px
    2024_FLUX ==>|" "| 2025_TileLink
    linkStyle 4 stroke:#FF69B4,stroke-width:2.5px
    2024_FLUX ==>|" "| 2026_FlashOverlap
    linkStyle 5 stroke:#00CED1,stroke-width:2.5px
    2024_Async_TP ==>|" "| 2025_TileLink
    linkStyle 6 stroke:#FFA500,stroke-width:2.5px
    2024_Async_TP ==>|" "| 2026_FlashOverlap
    linkStyle 7 stroke:#7B68EE,stroke-width:2.5px
    2025_NanoFlow ==>|" "| 2025_TokenWeave
    linkStyle 8 stroke:#9370DB,stroke-width:2.5px
```

## SVG Family

*5 methods, 4 relationships*

```mermaid
flowchart LR
    classDef defaultNode fill:#4A90E2,stroke:#2E5C8A,stroke-width:2px,color:#fff
    classDef rootNode fill:#50C878,stroke:#2E7D4E,stroke-width:3px,color:#fff
    classDef leafNode fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px,color:#fff
    linkStyle default stroke:#9370DB,stroke-width:2px

    2025_SVG2["SVG2[2025]"]
    class 2025_SVG2 leafNode
    2025_LiteAttention["LiteAttention[2025]"]
    class 2025_LiteAttention leafNode
    2025_RadialAttention["RadialAttention[2025]"]
    class 2025_RadialAttention rootNode
    2025_PAROAttention["PAROAttention[2025]"]
    class 2025_PAROAttention leafNode
    2025_SVG["SVG[2025]"]
    class 2025_SVG rootNode

    2025_RadialAttention ==>|" "| 2025_LiteAttention
    linkStyle 0 stroke:#9370DB,stroke-width:2.5px
    2025_SVG ==>|" "| 2025_SVG2
    linkStyle 1 stroke:#FF6347,stroke-width:2.5px
    2025_SVG ==>|" "| 2025_LiteAttention
    linkStyle 2 stroke:#20B2AA,stroke-width:2.5px
    2025_SVG ==>|" "| 2025_PAROAttention
    linkStyle 3 stroke:#FFD700,stroke-width:2.5px
```

## SparseGPT Family

*5 methods, 7 relationships*

```mermaid
flowchart LR
    classDef defaultNode fill:#4A90E2,stroke:#2E5C8A,stroke-width:2px,color:#fff
    classDef rootNode fill:#50C878,stroke:#2E7D4E,stroke-width:3px,color:#fff
    classDef leafNode fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px,color:#fff
    linkStyle default stroke:#9370DB,stroke-width:2px

    2023_SparseGPT["SparseGPT[2023]"]
    class 2023_SparseGPT rootNode
    2024_Wanda["Wanda[2024]"]
    class 2024_Wanda defaultNode
    2025_BaWA["BaWA[2025]"]
    class 2025_BaWA leafNode
    2025_SDS["SDS[2025]"]
    class 2025_SDS leafNode
    2024_Pruner_Zero["Pruner-Zero[2024]"]
    class 2024_Pruner_Zero leafNode

    2023_SparseGPT ==>|" "| 2025_SDS
    linkStyle 0 stroke:#9370DB,stroke-width:2.5px
    2023_SparseGPT ==>|" "| 2025_BaWA
    linkStyle 1 stroke:#FF6347,stroke-width:2.5px
    2023_SparseGPT ==>|" "| 2024_Wanda
    linkStyle 2 stroke:#20B2AA,stroke-width:2.5px
    2023_SparseGPT ==>|" "| 2024_Pruner_Zero
    linkStyle 3 stroke:#FFD700,stroke-width:2.5px
    2024_Wanda ==>|" "| 2025_SDS
    linkStyle 4 stroke:#FF69B4,stroke-width:2.5px
    2024_Wanda ==>|" "| 2025_BaWA
    linkStyle 5 stroke:#00CED1,stroke-width:2.5px
    2024_Wanda ==>|" "| 2024_Pruner_Zero
    linkStyle 6 stroke:#FFA500,stroke-width:2.5px
```

## 2024/StreamingLLM Family

*19 methods, 30 relationships*

```mermaid
flowchart LR
    classDef defaultNode fill:#4A90E2,stroke:#2E5C8A,stroke-width:2px,color:#fff
    classDef rootNode fill:#50C878,stroke:#2E7D4E,stroke-width:3px,color:#fff
    classDef leafNode fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px,color:#fff
    linkStyle default stroke:#9370DB,stroke-width:2px

    2024_streaming_llm["streaming-llm[2024]"]
    class 2024_streaming_llm rootNode
    2024_DoubleSparsity["DoubleSparsity[2024]"]
    class 2024_DoubleSparsity rootNode
    2024_Quest["Quest[2024]"]
    class 2024_Quest rootNode
    2025_BLASST["BLASST[2025]"]
    class 2025_BLASST leafNode
    2025_EvolKV["EvolKV[2025]"]
    class 2025_EvolKV leafNode
    2024_MInference["MInference[2024]"]
    class 2024_MInference defaultNode
    Ada_SnapKV["Ada-SnapKV"]
    class Ada_SnapKV rootNode
    2025_PureKV["PureKV[2025]"]
    class 2025_PureKV leafNode
    2025_XAttention["XAttention[2025]"]
    class 2025_XAttention defaultNode
    2024_SnapKV["SnapKV[2024]"]
    class 2024_SnapKV rootNode
    PyramidKV["PyramidKV"]
    class PyramidKV rootNode
    2023_H2O["H2O[2023]"]
    class 2023_H2O rootNode
    2025_TCA_Attention["TCA-Attention[2025]"]
    class 2025_TCA_Attention leafNode
    2024_InfLLM["InfLLM[2024]"]
    class 2024_InfLLM defaultNode
    2025_UNComp["UNComp[2025]"]
    class 2025_UNComp leafNode
    2024_StreamingLLM["2024/StreamingLLM"]
    class 2024_StreamingLLM rootNode
    CAKE["CAKE"]
    class CAKE rootNode
    2025_LAVa["LAVa[2025]"]
    class 2025_LAVa leafNode
    2025_FlexPrefill["FlexPrefill[2025]"]
    class 2025_FlexPrefill defaultNode

    2024_streaming_llm ==>|" "| 2025_XAttention
    linkStyle 0 stroke:#9370DB,stroke-width:2.5px
    2024_DoubleSparsity ==>|" "| 2025_UNComp
    linkStyle 1 stroke:#FF6347,stroke-width:2.5px
    2024_Quest ==>|" "| 2025_UNComp
    linkStyle 2 stroke:#20B2AA,stroke-width:2.5px
    2024_MInference ==>|" "| 2025_BLASST
    linkStyle 3 stroke:#FFD700,stroke-width:2.5px
    2024_MInference ==>|" "| 2025_FlexPrefill
    linkStyle 4 stroke:#FF69B4,stroke-width:2.5px
    2024_MInference ==>|" "| 2025_XAttention
    linkStyle 5 stroke:#00CED1,stroke-width:2.5px
    Ada_SnapKV ==>|" "| 2025_LAVa
    linkStyle 6 stroke:#FFA500,stroke-width:2.5px
    2025_XAttention ==>|" "| 2025_BLASST
    linkStyle 7 stroke:#7B68EE,stroke-width:2.5px
    2025_XAttention ==>|" "| 2025_TCA_Attention
    linkStyle 8 stroke:#9370DB,stroke-width:2.5px
    2024_SnapKV ==>|" "| 2025_UNComp
    linkStyle 9 stroke:#FF6347,stroke-width:2.5px
    2024_SnapKV ==>|" "| 2025_EvolKV
    linkStyle 10 stroke:#20B2AA,stroke-width:2.5px
    2024_SnapKV ==>|" "| 2025_LAVa
    linkStyle 11 stroke:#FFD700,stroke-width:2.5px
    2024_SnapKV ==>|" "| 2025_PureKV
    linkStyle 12 stroke:#FF69B4,stroke-width:2.5px
    PyramidKV ==>|" "| 2025_UNComp
    linkStyle 13 stroke:#00CED1,stroke-width:2.5px
    PyramidKV ==>|" "| 2025_EvolKV
    linkStyle 14 stroke:#FFA500,stroke-width:2.5px
    PyramidKV ==>|" "| 2025_LAVa
    linkStyle 15 stroke:#7B68EE,stroke-width:2.5px
    2023_H2O ==>|" "| 2025_UNComp
    linkStyle 16 stroke:#9370DB,stroke-width:2.5px
    2023_H2O ==>|" "| 2025_PureKV
    linkStyle 17 stroke:#FF6347,stroke-width:2.5px
    2023_H2O ==>|" "| 2024_InfLLM
    linkStyle 18 stroke:#20B2AA,stroke-width:2.5px
    2024_InfLLM ==>|" "| 2024_MInference
    linkStyle 19 stroke:#FFD700,stroke-width:2.5px
    2024_StreamingLLM ==>|" "| 2025_UNComp
    linkStyle 20 stroke:#FF69B4,stroke-width:2.5px
    2024_StreamingLLM ==>|" "| 2025_EvolKV
    linkStyle 21 stroke:#00CED1,stroke-width:2.5px
    2024_StreamingLLM ==>|" "| 2025_FlexPrefill
    linkStyle 22 stroke:#FFA500,stroke-width:2.5px
    2024_StreamingLLM ==>|" "| 2025_PureKV
    linkStyle 23 stroke:#7B68EE,stroke-width:2.5px
    2024_StreamingLLM ==>|" "| 2024_MInference
    linkStyle 24 stroke:#9370DB,stroke-width:2.5px
    2024_StreamingLLM ==>|" "| 2024_InfLLM
    linkStyle 25 stroke:#FF6347,stroke-width:2.5px
    CAKE ==>|" "| 2025_LAVa
    linkStyle 26 stroke:#20B2AA,stroke-width:2.5px
    2025_FlexPrefill ==>|" "| 2025_BLASST
    linkStyle 27 stroke:#FFD700,stroke-width:2.5px
    2025_FlexPrefill ==>|" "| 2025_XAttention
    linkStyle 28 stroke:#FF69B4,stroke-width:2.5px
    2025_FlexPrefill ==>|" "| 2025_TCA_Attention
    linkStyle 29 stroke:#00CED1,stroke-width:2.5px
```

## KIVI Family

*12 methods, 15 relationships*

```mermaid
flowchart LR
    classDef defaultNode fill:#4A90E2,stroke:#2E5C8A,stroke-width:2px,color:#fff
    classDef rootNode fill:#50C878,stroke:#2E7D4E,stroke-width:3px,color:#fff
    classDef leafNode fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px,color:#fff
    linkStyle default stroke:#9370DB,stroke-width:2px

    2025_KVmix["KVmix[2025]"]
    class 2025_KVmix leafNode
    2025_QJL["QJL[2025]"]
    class 2025_QJL leafNode
    2024_MiKV["MiKV[2024]"]
    class 2024_MiKV rootNode
    2024_KIVI["KIVI[2024]"]
    class 2024_KIVI rootNode
    Loki["Loki"]
    class Loki rootNode
    2025_H1B_KV["H1B-KV[2025]"]
    class 2025_H1B_KV leafNode
    2024_GEAR["GEAR[2024]"]
    class 2024_GEAR rootNode
    2025_RotateKV["RotateKV[2025]"]
    class 2025_RotateKV leafNode
    2025_VecInfer["VecInfer[2025]"]
    class 2025_VecInfer leafNode
    2025_MILLION["MILLION[2025]"]
    class 2025_MILLION defaultNode
    2024_KVQuant["KVQuant[2024]"]
    class 2024_KVQuant rootNode
    2024_ZipCache["ZipCache[2024]"]
    class 2024_ZipCache rootNode

    2024_MiKV ==>|" "| 2025_RotateKV
    linkStyle 0 stroke:#9370DB,stroke-width:2.5px
    2024_KIVI ==>|" "| 2025_KVmix
    linkStyle 1 stroke:#FF6347,stroke-width:2.5px
    2024_KIVI ==>|" "| 2025_MILLION
    linkStyle 2 stroke:#20B2AA,stroke-width:2.5px
    2024_KIVI ==>|" "| 2025_VecInfer
    linkStyle 3 stroke:#FFD700,stroke-width:2.5px
    2024_KIVI ==>|" "| 2025_QJL
    linkStyle 4 stroke:#FF69B4,stroke-width:2.5px
    2024_KIVI ==>|" "| 2025_H1B_KV
    linkStyle 5 stroke:#00CED1,stroke-width:2.5px
    2024_KIVI ==>|" "| 2025_RotateKV
    linkStyle 6 stroke:#FFA500,stroke-width:2.5px
    Loki ==>|" "| 2025_H1B_KV
    linkStyle 7 stroke:#7B68EE,stroke-width:2.5px
    2024_GEAR ==>|" "| 2025_RotateKV
    linkStyle 8 stroke:#9370DB,stroke-width:2.5px
    2025_MILLION ==>|" "| 2025_VecInfer
    linkStyle 9 stroke:#FF6347,stroke-width:2.5px
    2024_KVQuant ==>|" "| 2025_KVmix
    linkStyle 10 stroke:#20B2AA,stroke-width:2.5px
    2024_KVQuant ==>|" "| 2025_MILLION
    linkStyle 11 stroke:#FFD700,stroke-width:2.5px
    2024_KVQuant ==>|" "| 2025_QJL
    linkStyle 12 stroke:#FF69B4,stroke-width:2.5px
    2024_KVQuant ==>|" "| 2025_RotateKV
    linkStyle 13 stroke:#00CED1,stroke-width:2.5px
    2024_ZipCache ==>|" "| 2025_RotateKV
    linkStyle 14 stroke:#FFA500,stroke-width:2.5px
```

## NSA Family

*2 methods, 1 relationships*

```mermaid
flowchart LR
    classDef defaultNode fill:#4A90E2,stroke:#2E5C8A,stroke-width:2px,color:#fff
    classDef rootNode fill:#50C878,stroke:#2E7D4E,stroke-width:3px,color:#fff
    classDef leafNode fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px,color:#fff
    linkStyle default stroke:#9370DB,stroke-width:2px

    2025_NSA["NSA[2025]"]
    class 2025_NSA rootNode
    2025_InfLLM_V2["InfLLM-V2[2025]"]
    class 2025_InfLLM_V2 leafNode

    2025_NSA ==>|" "| 2025_InfLLM_V2
    linkStyle 0 stroke:#9370DB,stroke-width:2.5px
```
