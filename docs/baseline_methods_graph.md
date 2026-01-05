# Baseline Methods Graph

This page visualizes baseline-method relationships extracted from meta files.

Each component represents a family of related methods, showing how newer papers build upon previous baseline methods.

## StreamingLLM Family

*26 methods, 39 relationships*

```mermaid
flowchart LR
    classDef defaultNode fill:#4A90E2,stroke:#2E5C8A,stroke-width:2px,color:#fff
    classDef rootNode fill:#50C878,stroke:#2E7D4E,stroke-width:3px,color:#fff
    classDef leafNode fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px,color:#fff
    linkStyle default stroke:#9370DB,stroke-width:2px

    2023_H2O["H2O[2023]"]
    class 2023_H2O rootNode
    2024_AdaKV["AdaKV[2024]"]
    class 2024_AdaKV rootNode
    2024_DoubleSparsity["DoubleSparsity[2024]"]
    class 2024_DoubleSparsity rootNode
    2024_DuoAttention["DuoAttention[2024]"]
    class 2024_DuoAttention rootNode
    2024_InfLLM["InfLLM[2024]"]
    class 2024_InfLLM defaultNode
    2024_MInference["MInference[2024]"]
    class 2024_MInference defaultNode
    2024_Quest["Quest[2024]"]
    class 2024_Quest rootNode
    2024_SnapKV["SnapKV[2024]"]
    class 2024_SnapKV rootNode
    2024_StreamingLLM["StreamingLLM[2024]"]
    class 2024_StreamingLLM rootNode
    2025_BLASST["BLASST[2025]"]
    class 2025_BLASST leafNode
    2025_DefensiveKV["DefensiveKV[2025]"]
    class 2025_DefensiveKV leafNode
    2025_EvolKV["EvolKV[2025]"]
    class 2025_EvolKV leafNode
    2025_FlexPrefill["FlexPrefill[2025]"]
    class 2025_FlexPrefill defaultNode
    2025_FreeKV["FreeKV[2025]"]
    class 2025_FreeKV leafNode
    2025_Kascade["Kascade[2025]"]
    class 2025_Kascade leafNode
    2025_LAVa["LAVa[2025]"]
    class 2025_LAVa leafNode
    2025_PruLong["PruLong[2025]"]
    class 2025_PruLong leafNode
    2025_PureKV["PureKV[2025]"]
    class 2025_PureKV leafNode
    2025_RaaS["RaaS[2025]"]
    class 2025_RaaS rootNode
    2025_ShadowKV["ShadowKV[2025]"]
    class 2025_ShadowKV rootNode
    2025_TCA_Attention["TCA-Attention[2025]"]
    class 2025_TCA_Attention leafNode
    2025_UNComp["UNComp[2025]"]
    class 2025_UNComp leafNode
    2025_XAttention["XAttention[2025]"]
    class 2025_XAttention defaultNode
    Ada_SnapKV["Ada-SnapKV"]
    class Ada_SnapKV rootNode
    CAKE["CAKE"]
    class CAKE rootNode
    PyramidKV["PyramidKV"]
    class PyramidKV rootNode

    2023_H2O ==>|" "| 2024_InfLLM
    linkStyle 0 stroke:#9370DB,stroke-width:2.5px
    2023_H2O ==>|" "| 2025_PureKV
    linkStyle 1 stroke:#FF6347,stroke-width:2.5px
    2023_H2O ==>|" "| 2025_UNComp
    linkStyle 2 stroke:#20B2AA,stroke-width:2.5px
    2024_AdaKV ==>|" "| 2025_DefensiveKV
    linkStyle 3 stroke:#FFD700,stroke-width:2.5px
    2024_DoubleSparsity ==>|" "| 2025_UNComp
    linkStyle 4 stroke:#FF69B4,stroke-width:2.5px
    2024_DuoAttention ==>|" "| 2025_DefensiveKV
    linkStyle 5 stroke:#00CED1,stroke-width:2.5px
    2024_DuoAttention ==>|" "| 2025_PruLong
    linkStyle 6 stroke:#FFA500,stroke-width:2.5px
    2024_InfLLM ==>|" "| 2024_MInference
    linkStyle 7 stroke:#7B68EE,stroke-width:2.5px
    2024_MInference ==>|" "| 2025_BLASST
    linkStyle 8 stroke:#9370DB,stroke-width:2.5px
    2024_MInference ==>|" "| 2025_FlexPrefill
    linkStyle 9 stroke:#FF6347,stroke-width:2.5px
    2024_MInference ==>|" "| 2025_XAttention
    linkStyle 10 stroke:#20B2AA,stroke-width:2.5px
    2024_Quest ==>|" "| 2025_FreeKV
    linkStyle 11 stroke:#FFD700,stroke-width:2.5px
    2024_Quest ==>|" "| 2025_Kascade
    linkStyle 12 stroke:#FF69B4,stroke-width:2.5px
    2024_Quest ==>|" "| 2025_UNComp
    linkStyle 13 stroke:#00CED1,stroke-width:2.5px
    2024_SnapKV ==>|" "| 2025_DefensiveKV
    linkStyle 14 stroke:#FFA500,stroke-width:2.5px
    2024_SnapKV ==>|" "| 2025_EvolKV
    linkStyle 15 stroke:#7B68EE,stroke-width:2.5px
    2024_SnapKV ==>|" "| 2025_LAVa
    linkStyle 16 stroke:#9370DB,stroke-width:2.5px
    2024_SnapKV ==>|" "| 2025_PureKV
    linkStyle 17 stroke:#FF6347,stroke-width:2.5px
    2024_SnapKV ==>|" "| 2025_UNComp
    linkStyle 18 stroke:#20B2AA,stroke-width:2.5px
    2024_StreamingLLM ==>|" "| 2024_InfLLM
    linkStyle 19 stroke:#FFD700,stroke-width:2.5px
    2024_StreamingLLM ==>|" "| 2024_MInference
    linkStyle 20 stroke:#FF69B4,stroke-width:2.5px
    2024_StreamingLLM ==>|" "| 2025_EvolKV
    linkStyle 21 stroke:#00CED1,stroke-width:2.5px
    2024_StreamingLLM ==>|" "| 2025_FlexPrefill
    linkStyle 22 stroke:#FFA500,stroke-width:2.5px
    2024_StreamingLLM ==>|" "| 2025_Kascade
    linkStyle 23 stroke:#7B68EE,stroke-width:2.5px
    2024_StreamingLLM ==>|" "| 2025_PureKV
    linkStyle 24 stroke:#9370DB,stroke-width:2.5px
    2024_StreamingLLM ==>|" "| 2025_UNComp
    linkStyle 25 stroke:#FF6347,stroke-width:2.5px
    2024_StreamingLLM ==>|" "| 2025_XAttention
    linkStyle 26 stroke:#20B2AA,stroke-width:2.5px
    2025_FlexPrefill ==>|" "| 2025_BLASST
    linkStyle 27 stroke:#FFD700,stroke-width:2.5px
    2025_FlexPrefill ==>|" "| 2025_TCA_Attention
    linkStyle 28 stroke:#FF69B4,stroke-width:2.5px
    2025_FlexPrefill ==>|" "| 2025_XAttention
    linkStyle 29 stroke:#00CED1,stroke-width:2.5px
    2025_RaaS ==>|" "| 2025_FreeKV
    linkStyle 30 stroke:#FFA500,stroke-width:2.5px
    2025_ShadowKV ==>|" "| 2025_FreeKV
    linkStyle 31 stroke:#7B68EE,stroke-width:2.5px
    2025_XAttention ==>|" "| 2025_BLASST
    linkStyle 32 stroke:#9370DB,stroke-width:2.5px
    2025_XAttention ==>|" "| 2025_TCA_Attention
    linkStyle 33 stroke:#FF6347,stroke-width:2.5px
    Ada_SnapKV ==>|" "| 2025_LAVa
    linkStyle 34 stroke:#20B2AA,stroke-width:2.5px
    CAKE ==>|" "| 2025_LAVa
    linkStyle 35 stroke:#FFD700,stroke-width:2.5px
    PyramidKV ==>|" "| 2025_EvolKV
    linkStyle 36 stroke:#FF69B4,stroke-width:2.5px
    PyramidKV ==>|" "| 2025_LAVa
    linkStyle 37 stroke:#00CED1,stroke-width:2.5px
    PyramidKV ==>|" "| 2025_UNComp
    linkStyle 38 stroke:#FFA500,stroke-width:2.5px
```

## KIVI Family

*13 methods, 18 relationships*

```mermaid
flowchart LR
    classDef defaultNode fill:#4A90E2,stroke:#2E5C8A,stroke-width:2px,color:#fff
    classDef rootNode fill:#50C878,stroke:#2E7D4E,stroke-width:3px,color:#fff
    classDef leafNode fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px,color:#fff
    linkStyle default stroke:#9370DB,stroke-width:2px

    2024_GEAR["GEAR[2024]"]
    class 2024_GEAR rootNode
    2024_KIVI["KIVI[2024]"]
    class 2024_KIVI rootNode
    2024_KVQuant["KVQuant[2024]"]
    class 2024_KVQuant rootNode
    2024_MiKV["MiKV[2024]"]
    class 2024_MiKV rootNode
    2024_ZipCache["ZipCache[2024]"]
    class 2024_ZipCache rootNode
    2025_H1B_KV["H1B-KV[2025]"]
    class 2025_H1B_KV leafNode
    2025_KVmix["KVmix[2025]"]
    class 2025_KVmix leafNode
    2025_MILLION["MILLION[2025]"]
    class 2025_MILLION defaultNode
    2025_MixKVQ["MixKVQ[2025]"]
    class 2025_MixKVQ leafNode
    2025_QJL["QJL[2025]"]
    class 2025_QJL leafNode
    2025_RotateKV["RotateKV[2025]"]
    class 2025_RotateKV defaultNode
    2025_VecInfer["VecInfer[2025]"]
    class 2025_VecInfer leafNode
    Loki["Loki"]
    class Loki rootNode

    2024_GEAR ==>|" "| 2025_RotateKV
    linkStyle 0 stroke:#9370DB,stroke-width:2.5px
    2024_KIVI ==>|" "| 2025_H1B_KV
    linkStyle 1 stroke:#FF6347,stroke-width:2.5px
    2024_KIVI ==>|" "| 2025_KVmix
    linkStyle 2 stroke:#20B2AA,stroke-width:2.5px
    2024_KIVI ==>|" "| 2025_MILLION
    linkStyle 3 stroke:#FFD700,stroke-width:2.5px
    2024_KIVI ==>|" "| 2025_MixKVQ
    linkStyle 4 stroke:#FF69B4,stroke-width:2.5px
    2024_KIVI ==>|" "| 2025_QJL
    linkStyle 5 stroke:#00CED1,stroke-width:2.5px
    2024_KIVI ==>|" "| 2025_RotateKV
    linkStyle 6 stroke:#FFA500,stroke-width:2.5px
    2024_KIVI ==>|" "| 2025_VecInfer
    linkStyle 7 stroke:#7B68EE,stroke-width:2.5px
    2024_KVQuant ==>|" "| 2025_KVmix
    linkStyle 8 stroke:#9370DB,stroke-width:2.5px
    2024_KVQuant ==>|" "| 2025_MILLION
    linkStyle 9 stroke:#FF6347,stroke-width:2.5px
    2024_KVQuant ==>|" "| 2025_MixKVQ
    linkStyle 10 stroke:#20B2AA,stroke-width:2.5px
    2024_KVQuant ==>|" "| 2025_QJL
    linkStyle 11 stroke:#FFD700,stroke-width:2.5px
    2024_KVQuant ==>|" "| 2025_RotateKV
    linkStyle 12 stroke:#FF69B4,stroke-width:2.5px
    2024_MiKV ==>|" "| 2025_RotateKV
    linkStyle 13 stroke:#00CED1,stroke-width:2.5px
    2024_ZipCache ==>|" "| 2025_RotateKV
    linkStyle 14 stroke:#FFA500,stroke-width:2.5px
    2025_MILLION ==>|" "| 2025_VecInfer
    linkStyle 15 stroke:#7B68EE,stroke-width:2.5px
    2025_RotateKV ==>|" "| 2025_MixKVQ
    linkStyle 16 stroke:#9370DB,stroke-width:2.5px
    Loki ==>|" "| 2025_H1B_KV
    linkStyle 17 stroke:#FF6347,stroke-width:2.5px
```

## SVG Family

*8 methods, 7 relationships*

```mermaid
flowchart LR
    classDef defaultNode fill:#4A90E2,stroke:#2E5C8A,stroke-width:2px,color:#fff
    classDef rootNode fill:#50C878,stroke:#2E7D4E,stroke-width:3px,color:#fff
    classDef leafNode fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px,color:#fff
    linkStyle default stroke:#9370DB,stroke-width:2px

    2022_STA["STA[2022]"]
    class 2022_STA rootNode
    2024_SageAttention["SageAttention[2024]"]
    class 2024_SageAttention rootNode
    2025_FPSAttention["FPSAttention[2025]"]
    class 2025_FPSAttention leafNode
    2025_LiteAttention["LiteAttention[2025]"]
    class 2025_LiteAttention leafNode
    2025_PAROAttention["PAROAttention[2025]"]
    class 2025_PAROAttention leafNode
    2025_RadialAttention["RadialAttention[2025]"]
    class 2025_RadialAttention rootNode
    2025_SVG["SVG[2025]"]
    class 2025_SVG rootNode
    2025_SVG2["SVG2[2025]"]
    class 2025_SVG2 leafNode

    2022_STA ==>|" "| 2025_FPSAttention
    linkStyle 0 stroke:#9370DB,stroke-width:2.5px
    2024_SageAttention ==>|" "| 2025_FPSAttention
    linkStyle 1 stroke:#FF6347,stroke-width:2.5px
    2025_RadialAttention ==>|" "| 2025_LiteAttention
    linkStyle 2 stroke:#20B2AA,stroke-width:2.5px
    2025_SVG ==>|" "| 2025_FPSAttention
    linkStyle 3 stroke:#FFD700,stroke-width:2.5px
    2025_SVG ==>|" "| 2025_LiteAttention
    linkStyle 4 stroke:#FF69B4,stroke-width:2.5px
    2025_SVG ==>|" "| 2025_PAROAttention
    linkStyle 5 stroke:#00CED1,stroke-width:2.5px
    2025_SVG ==>|" "| 2025_SVG2
    linkStyle 6 stroke:#FFA500,stroke-width:2.5px
```

## PagedAttention Family

*8 methods, 10 relationships*

```mermaid
flowchart LR
    classDef defaultNode fill:#4A90E2,stroke:#2E5C8A,stroke-width:2px,color:#fff
    classDef rootNode fill:#50C878,stroke:#2E7D4E,stroke-width:3px,color:#fff
    classDef leafNode fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px,color:#fff
    linkStyle default stroke:#9370DB,stroke-width:2px

    2023_PagedAttention["PagedAttention[2023]"]
    class 2023_PagedAttention rootNode
    2024_Async_TP["Async-TP[2024]"]
    class 2024_Async_TP rootNode
    2024_FLUX["FLUX[2024]"]
    class 2024_FLUX rootNode
    2025_NanoFlow["NanoFlow[2025]"]
    class 2025_NanoFlow defaultNode
    2025_SparseServe["SparseServe[2025]"]
    class 2025_SparseServe leafNode
    2025_TileLink["TileLink[2025]"]
    class 2025_TileLink defaultNode
    2025_TokenWeave["TokenWeave[2025]"]
    class 2025_TokenWeave leafNode
    2026_FlashOverlap["FlashOverlap[2026]"]
    class 2026_FlashOverlap leafNode

    2023_PagedAttention ==>|" "| 2025_NanoFlow
    linkStyle 0 stroke:#9370DB,stroke-width:2.5px
    2023_PagedAttention ==>|" "| 2025_SparseServe
    linkStyle 1 stroke:#FF6347,stroke-width:2.5px
    2023_PagedAttention ==>|" "| 2025_TileLink
    linkStyle 2 stroke:#20B2AA,stroke-width:2.5px
    2023_PagedAttention ==>|" "| 2025_TokenWeave
    linkStyle 3 stroke:#FFD700,stroke-width:2.5px
    2024_Async_TP ==>|" "| 2025_TileLink
    linkStyle 4 stroke:#FF69B4,stroke-width:2.5px
    2024_Async_TP ==>|" "| 2026_FlashOverlap
    linkStyle 5 stroke:#00CED1,stroke-width:2.5px
    2024_FLUX ==>|" "| 2025_TileLink
    linkStyle 6 stroke:#FFA500,stroke-width:2.5px
    2024_FLUX ==>|" "| 2026_FlashOverlap
    linkStyle 7 stroke:#7B68EE,stroke-width:2.5px
    2025_NanoFlow ==>|" "| 2025_TokenWeave
    linkStyle 8 stroke:#9370DB,stroke-width:2.5px
    2025_TileLink ==>|" "| 2025_TokenWeave
    linkStyle 9 stroke:#FF6347,stroke-width:2.5px
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
    2024_Pruner_Zero["Pruner-Zero[2024]"]
    class 2024_Pruner_Zero leafNode
    2024_Wanda["Wanda[2024]"]
    class 2024_Wanda defaultNode
    2025_BaWA["BaWA[2025]"]
    class 2025_BaWA leafNode
    2025_SDS["SDS[2025]"]
    class 2025_SDS leafNode

    2023_SparseGPT ==>|" "| 2024_Pruner_Zero
    linkStyle 0 stroke:#9370DB,stroke-width:2.5px
    2023_SparseGPT ==>|" "| 2024_Wanda
    linkStyle 1 stroke:#FF6347,stroke-width:2.5px
    2023_SparseGPT ==>|" "| 2025_BaWA
    linkStyle 2 stroke:#20B2AA,stroke-width:2.5px
    2023_SparseGPT ==>|" "| 2025_SDS
    linkStyle 3 stroke:#FFD700,stroke-width:2.5px
    2024_Wanda ==>|" "| 2024_Pruner_Zero
    linkStyle 4 stroke:#FF69B4,stroke-width:2.5px
    2024_Wanda ==>|" "| 2025_BaWA
    linkStyle 5 stroke:#00CED1,stroke-width:2.5px
    2024_Wanda ==>|" "| 2025_SDS
    linkStyle 6 stroke:#FFA500,stroke-width:2.5px
```

## GPTQ Family

*2 methods, 1 relationships*

```mermaid
flowchart LR
    classDef defaultNode fill:#4A90E2,stroke:#2E5C8A,stroke-width:2px,color:#fff
    classDef rootNode fill:#50C878,stroke:#2E7D4E,stroke-width:3px,color:#fff
    classDef leafNode fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px,color:#fff
    linkStyle default stroke:#9370DB,stroke-width:2px

    2023_GPTQ["GPTQ[2023]"]
    class 2023_GPTQ rootNode
    2025_SQ_format["SQ-format[2025]"]
    class 2025_SQ_format leafNode

    2023_GPTQ ==>|" "| 2025_SQ_format
    linkStyle 0 stroke:#9370DB,stroke-width:2.5px
```

## DHC Family

*2 methods, 1 relationships*

```mermaid
flowchart LR
    classDef defaultNode fill:#4A90E2,stroke:#2E5C8A,stroke-width:2px,color:#fff
    classDef rootNode fill:#50C878,stroke:#2E7D4E,stroke-width:3px,color:#fff
    classDef leafNode fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px,color:#fff
    linkStyle default stroke:#9370DB,stroke-width:2px

    2025_DHC["DHC[2025]"]
    class 2025_DHC rootNode
    2025_mHC["mHC[2025]"]
    class 2025_mHC leafNode

    2025_DHC ==>|" "| 2025_mHC
    linkStyle 0 stroke:#9370DB,stroke-width:2.5px
```

## NSA Family

*2 methods, 1 relationships*

```mermaid
flowchart LR
    classDef defaultNode fill:#4A90E2,stroke:#2E5C8A,stroke-width:2px,color:#fff
    classDef rootNode fill:#50C878,stroke:#2E7D4E,stroke-width:3px,color:#fff
    classDef leafNode fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px,color:#fff
    linkStyle default stroke:#9370DB,stroke-width:2px

    2025_InfLLM_V2["InfLLM-V2[2025]"]
    class 2025_InfLLM_V2 leafNode
    2025_NSA["NSA[2025]"]
    class 2025_NSA rootNode

    2025_NSA ==>|" "| 2025_InfLLM_V2
    linkStyle 0 stroke:#9370DB,stroke-width:2.5px
```
