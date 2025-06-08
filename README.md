# odiga_backend


sequenceDiagram
    participant User
    participant SystemCoordinator
    participant ImageAnalyzer
    participant ContentCreator
    participant MultimodalAgent
    participant TemplateSelector
    participant JSXGenerator
    participant MagazineDB
    
    User->>SystemCoordinator: Request Magazine Generation
    
    %% Phase 1
    Note over SystemCoordinator,JSXGenerator: Phase 1: Image Analysis & Content Generation
    SystemCoordinator->>ImageAnalyzer: Analyze Images
    ImageAnalyzer-->>SystemCoordinator: Return Image Analysis Results
    SystemCoordinator->>MagazineDB: Save Image Analysis
    
    SystemCoordinator->>ContentCreator: Generate Content
    
    %% ContentCreator internal flow
    Note over ContentCreator: Internal parallel execution:<br/>1. Interview Processing<br/>2. Essay Processing
    Note over ContentCreator: Sequential steps after parallel completion:<br/>1. Content Structure Planning<br/>2. Section Content Generation<br/>3. Content Refinement
    
    ContentCreator-->>SystemCoordinator: Return Raw Content JSON
    SystemCoordinator->>MagazineDB: Save Raw Content (phase1_completed)
    
    %% Phase 2
    Note over SystemCoordinator,JSXGenerator: Phase 2: Unified Multimodal Processing
    SystemCoordinator->>MultimodalAgent: Process Magazine Unified
    
    %% MultimodalAgent internal flow
    Note over MultimodalAgent: Sequential processing:<br/>1. Semantic Analysis<br/>2. AI Search Pattern Collection<br/>3. CrewAI Tasks (Content Structure, Image Layout, Semantic Coordination)
    
    MultimodalAgent-->>SystemCoordinator: Return Unified Results
    SystemCoordinator->>MagazineDB: Update Magazine (phase2_completed)
    
    %% Phase 3
    Note over SystemCoordinator,JSXGenerator: Phase 3: Template Selection & JSX Preparation
    
    SystemCoordinator->>TemplateSelector: Analyze and Select Templates
    TemplateSelector-->>SystemCoordinator: Return Selected Templates
    
    %% Phase 4
    Note over SystemCoordinator,JSXGenerator: Phase 4: Final JSX Assembly
    
    SystemCoordinator->>JSXGenerator: Generate JSX for Each Section
    JSXGenerator-->>SystemCoordinator: Return Final JSX Components
    
    SystemCoordinator->>MagazineDB: Update Magazine (completed)
    
    SystemCoordinator-->>User: Return Magazine Generation Result
