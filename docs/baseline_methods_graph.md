# Baseline Methods Graph

This page visualizes baseline-method relationships extracted from meta files.

## Component 1

```mermaid
flowchart TD
    2025_BaWA["BaWA[2025]"]
    2024_Wanda["Wanda[2024]"]
    2025_SDS["SDS[2025]"]
    2023_sparsegpt["sparsegpt[2023]"]
    2024_Wanda --> 2025_SDS
    2024_Wanda --> 2025_BaWA
    2023_sparsegpt --> 2025_SDS
    2023_sparsegpt --> 2025_BaWA
    2023_sparsegpt --> 2024_Wanda
```

## Component 2

```mermaid
flowchart TD
    2023_PagedAttention["PagedAttention[2023]"]
    2025_TileLink["TileLink[2025]"]
    2025_TokenWeave["TokenWeave[2025]"]
    2025_NanoFlow["NanoFlow[2025]"]
    2023_PagedAttention --> 2025_NanoFlow
    2023_PagedAttention --> 2025_TokenWeave
    2025_TileLink --> 2025_TokenWeave
    2025_NanoFlow --> 2025_TokenWeave
```

## Component 3

```mermaid
flowchart TD
    2025_XAttention["XAttention[2025]"]
    2024_streaming_llm["streaming-llm[2024]"]
    2024_MInference["MInference[2024]"]
    2025_FlexPrefill["FlexPrefill[2025]"]
    2024_streaming_llm --> 2025_FlexPrefill
    2024_streaming_llm --> 2025_XAttention
    2024_MInference --> 2025_FlexPrefill
    2024_MInference --> 2025_XAttention
    2025_FlexPrefill --> 2025_XAttention
```
