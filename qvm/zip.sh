#!/bin/bash

# Nome do arquivo zipado
ZIP_NAME="projeto.zip"

# Verifica se os arquivos/diretórios existem antes de zipar
if [[ -d "src" && -d "tests" && -f "Cargo.toml" ]]; then
    zip -r "$ZIP_NAME" src tests Cargo.toml
    echo "Arquivos zipados com sucesso em $ZIP_NAME"
else
    echo "Erro: Um ou mais arquivos/diretórios não encontrados."
    exit 1
fi
