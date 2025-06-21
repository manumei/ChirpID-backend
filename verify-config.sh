#!/bin/bash

echo "🔍 Verificando configuración ChirpID Backend..."
echo "=============================================="

# Verificar que no hay archivos nginx
if find . -name "*nginx*" -type f -o -name "*nginx*" -type d | grep -q nginx; then
    echo "❌ ADVERTENCIA: Se encontraron archivos/directorios nginx residuales:"
    find . -name "*nginx*" -type f -o -name "*nginx*" -type d
else
    echo "✅ No hay archivos nginx residuales"
fi

# Verificar docker-compose.yml
if [ -f "docker-compose.yml" ]; then
    echo "✅ docker-compose.yml existe"
    
    # Verificar que no hay referencias nginx
    if grep -q nginx docker-compose.yml; then
        echo "❌ ADVERTENCIA: docker-compose.yml contiene referencias a nginx"
        grep -n nginx docker-compose.yml
    else
        echo "✅ docker-compose.yml no contiene referencias a nginx"
    fi
    
    # Verificar red externa
    if grep -q "app-network" docker-compose.yml && grep -q "external: true" docker-compose.yml; then
        echo "✅ docker-compose.yml usa red externa app-network"
    else
        echo "❌ docker-compose.yml no está configurado para red externa"
    fi
    
    # Verificar puerto exposed
    if grep -q "expose:" docker-compose.yml; then
        echo "✅ Backend expone puerto correctamente"
    else
        echo "❌ Backend no expone puerto"
    fi
    
    # Verificar healthcheck
    if grep -q "healthcheck:" docker-compose.yml; then
        echo "✅ Backend tiene healthcheck configurado"
    else
        echo "⚠️ Backend no tiene healthcheck"
    fi
else
    echo "❌ docker-compose.yml no encontrado"
fi

# Verificar workflow
if [ -f ".github/workflows/deploy.yml" ]; then
    echo "✅ Workflow de deploy existe"
    
    # Verificar que no hay referencias nginx
    if grep -q nginx .github/workflows/deploy.yml; then
        echo "❌ ADVERTENCIA: Workflow contiene referencias a nginx"
        grep -n nginx .github/workflows/deploy.yml
    else
        echo "✅ Workflow no contiene referencias a nginx"
    fi
    
    # Verificar self-hosted runner
    if grep -q "self-hosted" .github/workflows/deploy.yml; then
        echo "✅ Workflow usa self-hosted runner"
    else
        echo "⚠️ Workflow no usa self-hosted runner"
    fi
    
    # Verificar tests
    if grep -q "^  test:" .github/workflows/deploy.yml; then
        echo "✅ Workflow incluye tests"
    else
        echo "⚠️ Workflow no incluye tests"
    fi
else
    echo "❌ Workflow de deploy no encontrado"
fi

echo ""
echo "🎯 Resumen:"
echo "- Este backend debe solo exponer su servicio en la red app-network"
echo "- El nginx central (server-nginx) se encarga del proxy y certificados"
echo "- El workflow debe solo construir y deployar el backend"
echo "- Los tests se ejecutan antes del deploy"
echo ""
