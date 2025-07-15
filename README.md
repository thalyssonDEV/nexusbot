# NexusBot 🤖

NexusBot é um projeto de chatbot inteligente e multifuncional, desenvolvido com Python e o framework FastAPI. Ele utiliza o poder do modelo de IA generativa `gemini-1.5-flash-latest` da Google para fornecer respostas coesas e contextuais, tanto para texto quanto para imagens.

## 🔗 Aplicação Online

Acesse a demonstração ao vivo da aplicação através do link:
**[https://nexusbot.zapto.org](https://nexusbot.zapto.org)**

## ✨ Recursos Principais

- **IA Generativa Avançada**: Utiliza o modelo Gemini 1.5 Flash da Google para interações inteligentes.
- **Suporte Multimodal**: Processa tanto texto quanto imagens.
- **Histórico de Conversa**: Gerenciamento de sessões com **Redis** para manter o contexto das conversas.
- **Interface Moderna**: UI responsiva com **TailwindCSS**, incluindo tema claro/escuro e suporte a múltiplos idiomas.
- **Implantação Segura e Escalável**: Executado em uma VM do Google Cloud com **Docker**, **Nginx** (proxy reverso) e **Let's Encrypt** (SSL automático) para HTTPS.

## 🛠️ Tecnologias Utilizadas

- **Backend**: Python, FastAPI
- **Inteligência Artificial**: Google Generative AI (Gemini)
- **Banco de Dados/Cache**: Redis
- **Frontend**: HTML, JavaScript, TailwindCSS
- **Containerização**: Docker, Docker Compose
- **Servidor Web/Proxy**: Nginx, Gunicorn, Let's Encrypt (para HTTPS)
- **Cloud**: Google Cloud (VM)

## 📄 Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
