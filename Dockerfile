# Usar uma imagem base oficial do Python.
# python:3.13-slim é uma boa escolha para um tamanho de imagem menor.
FROM python:3.13-slim

# Definir variáveis de ambiente para evitar que o Python gere ficheiros .pyc e para o buffer de saída.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Definir o diretório de trabalho dentro do contentor.clea
WORKDIR /app

# Copiar o ficheiro de requisitos primeiro para aproveitar o cache do Docker.
# Se requirements.txt não mudar, esta camada não será reconstruída.
COPY requirements.txt .

# Instalar as dependências do projeto.
# --no-cache-dir para reduzir o tamanho da imagem.
# --upgrade pip para garantir que temos a versão mais recente do pip.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar todo o código da aplicação para o diretório de trabalho /app.
COPY . .

# Expor a porta em que a aplicação Uvicorn será executada (o padrão é 8000).
EXPOSE 8000

# Comando para executar a aplicação Uvicorn.
# Substitua "main" pelo nome do seu ficheiro Python principal se for diferente.
# Substitua "app" pelo nome da sua instância FastAPI se for diferente.
# O host 0.0.0.0 torna a aplicação acessível de fora do contentor.
# O Gunicorn irá gerir os processos de trabalho do Uvicorn.
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "main:app"]