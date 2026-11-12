# Arquitetura e Guia de Publicação para Demonstração Online (com BYOK)
> **Lexicon: Context-Aware Book Translation Pipeline with Lexical Consensus**

Este documento detalha o plano técnico para empacotar e publicar o **Lexicon** como uma aplicação de demonstração web interativa (*live demo*), projetada para recrutadores, avaliadores técnicos e a comunidade de IA Aplicada.

---

## 1. Visão Geral e Experiência do Usuário (UX em 90 Segundos)

Um Tech Lead ou recrutador tem pouco tempo para avaliar um projeto de portfólio. A aplicação não deve forçar o usuário a esperar meia hora por 80 capítulos, mas sim **demonstrar visualmente o mecanismo de consenso e as métricas em tempo real**.

A aplicação web oferecerá **3 modos de operação**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        MODOS DA DEMONSTRAÇÃO WEB                       │
├──────────────────────────┬──────────────────────────┬──────────────────┤
│ 1. Guided Tour (Oz)      │ 2. Live Sandbox          │ 3. Modo BYOK     │
│ (Custo $0 / Instantâneo) │ (Upload até 3 capítulos) │ (Sua Própria Key)│
│                          │                          │                  │
│ • Corpus público (Oz)    │ • Upload EPUB/Markdown   │ • Sem limite     │
│ • Resposta em cache ou   │ • Limitado a 3 capítulos │ • Claude, GPT-4o │
│   Gemini Flash (< 3s)    │ • Chave do host (grátis) │ • Chave efêmera  │
└──────────────────────────┴──────────────────────────┴──────────────────┘
```

---

## 2. Arquitetura do Sistema Web (Event-Driven Streaming)

Chamadas a LLMs para múltiplos capítulos em paralelo demoram entre 3 a 15 segundos. Um endpoint HTTP tradicional (`POST` síncrono) arriscaria *timeout* de proxies (Cloudflare, Vercel).

A solução é uma arquitetura assíncrona orientada a eventos usando **Server-Sent Events (SSE)** ou **WebSockets**:

```
[ Frontend: React / Next.js ou Streamlit ]
                   │
                   ▼  HTTP POST /api/translate/stream (com SSE)
[ Backend: FastAPI Assíncrono ]
         │
         ├── Middleware de Segurança:
         │   ├── Rate Limiter (SlowAPI por IP)
         │   ├── Input Guardrails (Max 3 capítulos, sanitização XSS)
         │   └── Key Resolver (Host Key ou BYOK do Header)
         │
         ▼
[ Lexicon Orchestrator (Asyncio, LangGraph ou Google ADK) ]
         │
         ├── Evento 1: {"phase": 1, "status": "snapshot_created", "version": 1}
         ├── Evento 2: {"phase": 2, "worker": "ch_1", "status": "translated", "proposal": "gorrete de ouro"}
         ├── Evento 3: {"phase": 2, "worker": "ch_2", "status": "translated", "proposal": "chapéu dourado"}
         ├── Evento 4: {"phase": 3, "conflict": "golden cap", "winner": "gorrete de ouro", "affected": [2, 3]}
         ├── Evento 5: {"phase": 4, "reconciled": [2, 3], "bypass": [1]}
         └── Evento 6: {"phase": 5, "metrics": {"GAR": 100.0, "entropy_bits": 0.0, "chrf": 100.0}}
```

---

## 3. Especificação do Modo BYOK (*Bring Your Own Key*)

O modo BYOK permite que visitantes testem livros maiores ou modelos caros sem onerar a sua conta de desenvolvedor.

### 3.1. Princípios de Segurança do BYOK
1. **Chave 100% Efêmera**: A chave do usuário **NUNCA** é salva em banco de dados, arquivos de log ou disco.
2. **Armazenamento no Cliente**: Salva apenas no `sessionStorage` do navegador do visitante (limpa ao fechar a aba).
3. **Transporte Seguro**: Trafega exclusivamente via cabeçalhos HTTPS criptografados:
   - `X-API-Key: sk-ant-...` ou `X-API-Key: AIzaSy...`
   - `X-Model-Provider: anthropic | openai | google | groq | openrouter`
4. **Descarte em Memória**: O backend instancia o `LLMProvider` sob demanda e descarta o objeto ao encerrar a requisição SSE.

### 3.2. Provedores Suportados no BYOK
Graças à Clean Architecture do Lexicon (`OpenAICompatibleProvider`), o suporte a provedores adicionais é imediato:
* **Google Gemini API**: `gemini-2.0-flash` (ultrarrápido e custo ínfimo).
* **Groq Cloud**: `llama-3.3-70b-versatile` (~250 tokens/segundo).
* **OpenAI**: `gpt-4o`, `gpt-4o-mini`.
* **Anthropic**: `claude-3-5-sonnet-20241022`, `claude-3-5-haiku`.
* **OpenRouter**: Acesso a qualquer modelo aberto ou proprietário.

---

## 4. Proteção de Custos e Guardrails de Produção

Em uma demonstração pública, proteções contra abuso comprovam maturidade de Engenharia de Software:

1. **Camada de Cache Determinístico**:
   - Para o exemplo padrão de *The Wonderful Wizard of Oz*, o backend possui as respostas pré-calculadas. Se o visitante clicar em *"Executar Tour Guiado"*, o custo é **exatamente \$0.00**.
2. **Rate Limiting Estrito**:
   - `5 requisições / hora por IP` no modo gratuito.
   - Ilimitado para requisições com chave BYOK válida.
3. **Travas Rígidas de Entrada (Payload Capping)**:
   - Tamanho máximo de arquivo: `2 MB`.
   - Contagem máxima de capítulos no modo grátis: `3 capítulos`.
   - Sanitização de Markdown com `bleach` contra ataques de XSS no renderizador HTML.

---

## 5. Design da Interface do Usuário (UI/UX)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ LEXICON: Context-Aware Book Translation with Lexical Consensus              │
├──────────────────────────────┬──────────────────────────────────────────────┤
│ PAINEL DE CONTROLE           │ FLUXO DE EXECUÇÃO EM TEMPO REAL (STREAMING)   │
│                              │                                              │
│ 1. Motor de Orquestração:    │ [Fase 1: Snapshot Isolation]                 │
│    ( ) Pure Asyncio          │  ✓ Snapshot v1 congelado (3 termos ativos)    │
│    (*) LangGraph StateGraph  │                                              │
│    ( ) Google ADK            │ [Fase 2: Map Workers Paralelos]              │
│                              │  ✓ Cap. 1: Proposta 'gorrete de ouro'         │
│ 2. Provedor & Modelo:        │  ✓ Cap. 2: Proposta 'chapéu dourado'         │
│    [Gemini 2.0 Flash      v] │  ✓ Cap. 3: Proposta 'coroa áurea'            │
│                              │                                              │
│ 3. Modo de Execução:         │ [Fase 3: Single-Writer Reduce]               │
│    (*) Demo Oz (Grátis)      │  ⚠️ Conflito: 'golden cap' (3 propostas)     │
│    ( ) Upload EPUB           │  ✓ Vencedor: 'gorrete de ouro' (Cap. 1 min)   │
│                              │                                              │
│ 4. BYOK (Opcional):          │ [Fase 4: Reconciliação Seletiva]             │
│    [sk-ant-...            ]  │  ✓ Cap. 2 revisado ('chapéu' -> 'gorrete')   │
│                              │  ✓ Cap. 3 revisado ('da coroa' -> 'do gorrete')│
│    [  EXECUTAR PIPELINE  ]   │  ✓ Cap. 1 em bypass (0 tokens extras)        │
│                              │                                              │
│                              │ [Quality Gate: Métricas Finais]              │
│                              │  • Aderência: 100% | Entropia H: 0.0000 b    │
│                              │  • Gênero: 100%   | chrF Score: 100.0        │
└──────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 6. Opções de Hospedagem (Custo Zero a Mínimo)

| Opção | Stack Técnica | Custo Mensal | Vantagens para o Portfólio |
| :--- | :--- | :--- | :--- |
| **Opção A: Streamlit Community ou Hugging Face Spaces** | Python + Streamlit em Docker | **\$0.00** | Rápido de construir; repositório único; HTTPS e container inclusos. |
| **Opção B: Vercel + Railway / Render** | Frontend Next.js + Backend FastAPI | **\$0.00 – \$5.00** | Visual moderno de SaaS profissional; latência ultrabaixa com Edge network. |
| **Opção C: Static Showcase + Terminal Interativo** | GitHub Pages (HTML/JS) + Asciinema player | **\$0.00** | Zero infraestrutura de backend; reproduz a CLI real sem custos de servidor. |

---

## 7. Roteiro Passo a Passo para Publicação

Quando você decidir colocar a demo online, o caminho de implementação recomendado é:

1. **Etapa 1: Endpoint FastAPI (`src/lexicon/server.py`)**
   - Criar endpoint `POST /api/translate/stream` retornando `StreamingResponse(generator, media_type="text/event-stream")`.
   - Adicionar middleware de validação de BYOK e limitação de capítulos.
2. **Etapa 2: Interface Web**
   - Criar frontend conectado aos eventos SSE para acender os cards das fases dinamicamente.
3. **Etapa 3: Deploy & Links**
   - Configurar variáveis de ambiente no serviço de hospedagem (`GEMINI_API_KEY` ou `GROQ_API_KEY` para o modo público).
   - Adicionar badge no topo do [`README.md`](../README.md) e [`README.en.md`](../README.en.md):
     ```markdown
     [![Live Demo](https://img.shields.io/badge/Live%20Demo-Lexicon%20Web-blue?style=for-the-badge&logo=googlechrome)](https://lexicon-demo.example.com)
     ```
