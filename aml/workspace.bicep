// Azure Machine Learning Workspace
// Plantilla Bicep para crear workspace con dependencias

@description('Nombre del workspace de Azure ML')
param workspaceName string

@description('Ubicación de los recursos')
param location string = resourceGroup().location

@description('Environment tag (dev, test, prod)')
param environment string = 'dev'

@description('Tags adicionales para los recursos')
param tags object = {
  Environment: environment
  Project: 'data-agent-pro'
  ManagedBy: 'bicep'
}

// Storage Account para el workspace
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: 'st${uniqueString(resourceGroup().id)}ml'
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    encryption: {
      services: {
        blob: {
          enabled: true
        }
        file: {
          enabled: true
        }
      }
      keySource: 'Microsoft.Storage'
    }
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
  }
}

// Key Vault para secretos
resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' = {
  name: 'kv-${uniqueString(resourceGroup().id)}-ml'
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enabledForDeployment: true
    enabledForDiskEncryption: true
    enabledForTemplateDeployment: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    enablePurgeProtection: true
    accessPolicies: []
  }
}

// Application Insights para monitoring
resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-${uniqueString(resourceGroup().id)}-ml'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

// Container Registry para environments
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: 'cr${uniqueString(resourceGroup().id)}ml'
  location: location
  tags: tags
  sku: {
    name: 'Standard'
  }
  properties: {
    adminUserEnabled: true
    publicNetworkAccess: 'Enabled'
  }
}

// Azure ML Workspace
resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2023-04-01' = {
  name: workspaceName
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: workspaceName
    description: 'Azure ML Workspace for ${environment} environment'
    storageAccount: storageAccount.id
    keyVault: keyVault.id
    applicationInsights: applicationInsights.id
    containerRegistry: containerRegistry.id
    publicNetworkAccess: 'Enabled'
    hbiWorkspace: false
  }
}

// Outputs para usar en otros scripts
output workspaceId string = mlWorkspace.id
output workspaceName string = mlWorkspace.name
output storageAccountId string = storageAccount.id
output keyVaultId string = keyVault.id
output applicationInsightsId string = applicationInsights.id
output containerRegistryId string = containerRegistry.id
