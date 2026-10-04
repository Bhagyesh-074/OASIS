# 2. Related Work

The literature on multi-agent LLM systems between 2023 and 2025 spans twelve primary systems, categorized below by architectural philosophy.

### 2.1 Agent Orchestration & Execution Frameworks
- **LangGraph** (2024) models multi-agent execution as cyclical state machines. While LangGraph provides robust checkpointing and manual human-in-the-loop intervention, it lacks autonomous budget governance, model admission checks, and automated node replacement.
- **AutoGen / AG2** (Wu et al., 2024) formalizes conversational agent programming. AutoGen allows setting maximum conversational turns, but lacks multi-dimensional admission control or in-flight quality-driven agent swapping.
- **CrewAI** (2024) organizes agents into role-playing teams with sequential or hierarchical execution, but treats model costs and runtime errors as external unmanaged exceptions.

### 2.2 Role-Based & Software Engineering Systems
- **MetaGPT** (Hong et al., 2024) embeds Standard Operating Procedures (SOPs) into multi-agent workflows. Its static role pipeline prevents dynamic team rightsizing based on task complexity.
- **ChatDev** (Qian et al., 2024) simulates virtual software teams using chat-based chains. ChatDev addresses hallucinations through communicative peer discussions, which incurs quadratic token consumption and accelerates financial budget exhaustion.

### 2.3 Dynamic Sizing, Reconfiguration & Governance
- **AgentVerse** (Chen et al., 2024) and **DyLAN** (Liu et al., 2023) introduce dynamic team formation and agent importance scoring. However, both frameworks rely on dense peer-to-peer LLM rating matrices, creating severe supervisory overhead that often exceeds the cost of task execution itself.
- **CaptainAgent** (Wang et al., 2024) delegates team sizing to a recursive "captain" agent. This unconstrained recursive generation risks runaway agent expansion.
- **MacNet** (Wang et al., 2024), **CAMEL** (Li et al., 2023), **AutoDefense** (Yao et al., 2024), and **GPTSwarm** (Zhuge et al., 2024) explore topological routing, communicative cooperation, safety defense, and RL graph mutation, but none provide unified 4D budget enforcement coupled with layered quality gating.

### 2.4 The OASIS Differentiation
OASIS does not compete on framework ergonomics. Instead, OASIS functions as an external supervisory layer decoupling governance from agent code. As summarized in Table 1, OASIS uniquely unifies pre-execution team sizing, deterministic 4D admission control, tiered L1–L3 monitoring, in-flight agent substitution, and verified replay purity.
