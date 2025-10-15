// Azure ML Compute Clusters
// Plantilla Bicep para crear compute targets

@description('Nombre del workspace de Azure ML')
param workspaceName string

@description('Ubicación de los recursos')
param location string = resourceGroup().location

@description('Environment tag')
param environment string = 'dev'

// Referencia al workspace existente
resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2023-04-01' existing = {
  name: workspaceName
}

// CPU Compute Cluster (general purpose)
resource cpuCluster 'Microsoft.MachineLearningServices/workspaces/computes@2023-04-01' = {
  parent: mlWorkspace
  name: 'cpu-cluster'
  location: location
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: 'Standard_DS3_v2'
      vmPriority: 'Dedicated'
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: environment == 'prod' ? 10 : 4
        nodeIdleTimeBeforeScaleDown: 'PT120S'
      }
      remoteLoginPortPublicAccess: 'Disabled'
      osType: 'Linux'
    }
  }
}

// GPU Compute Cluster (for deep learning)
resource gpuCluster 'Microsoft.MachineLearningServices/workspaces/computes@2023-04-01' = {
  parent: mlWorkspace
  name: 'gpu-cluster'
  location: location
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: 'Standard_NC6'
      vmPriority: environment == 'prod' ? 'Dedicated' : 'LowPriority'
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: environment == 'prod' ? 4 : 2
        nodeIdleTimeBeforeScaleDown: 'PT300S'
      }
      remoteLoginPortPublicAccess: 'Disabled'
      osType: 'Linux'
    }
  }
}

// Compute Instance (for development)
resource computeInstance 'Microsoft.MachineLearningServices/workspaces/computes@2023-04-01' = if (environment == 'dev') {
  parent: mlWorkspace
  name: 'dev-instance'
  location: location
  properties: {
    computeType: 'ComputeInstance'
    properties: {
      vmSize: 'Standard_DS3_v2'
      applicationSharingPolicy: 'Personal'
      sshSettings: {
        sshPublicAccess: 'Disabled'
      }
    }
  }
}

output cpuClusterId string = cpuCluster.id
output gpuClusterId string = gpuCluster.id
output computeInstanceId string = environment == 'dev' ? computeInstance.id : ''
