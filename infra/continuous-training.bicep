targetScope = 'resourceGroup'

@description('Existing Azure Machine Learning workspace name.')
param workspaceName string

@description('Existing storage account used for continuous training data.')
param storageAccountName string

@description('Blob container created for immutable dataset snapshots.')
param containerName string = 'yolo-training'

@description('Object ID of the GitHub OIDC service principal, not its client ID.')
param githubPrincipalId string

@description('Existing Azure Machine Learning CPU compute cluster name.')
param cpuComputeName string

resource workspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' existing = {
  name: workspaceName
}

resource cpuCompute 'Microsoft.MachineLearningServices/workspaces/computes@2024-04-01' existing = {
  parent: workspace
  name: cpuComputeName
}

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageAccountName
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' existing = {
  parent: storageAccount
  name: 'default'
}

resource trainingContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: containerName
  properties: {
    publicAccess: 'None'
  }
}

var storageBlobDataContributorRoleId = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
)
var storageBlobDataReaderRoleId = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1'
)
var azureMlDataScientistRoleId = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  'f6c7c914-8db3-469d-8ca1-694a8f32e121'
)

resource githubStorageRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageAccount.id, githubPrincipalId, storageBlobDataContributorRoleId)
  scope: storageAccount
  properties: {
    principalId: githubPrincipalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: storageBlobDataContributorRoleId
  }
}

resource githubWorkspaceRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(workspace.id, githubPrincipalId, azureMlDataScientistRoleId)
  scope: workspace
  properties: {
    principalId: githubPrincipalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: azureMlDataScientistRoleId
  }
}

resource workspaceStorageRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageAccount.id, workspace.id, storageBlobDataReaderRoleId)
  scope: storageAccount
  properties: {
    principalId: workspace.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: storageBlobDataReaderRoleId
  }
}

resource computeStorageRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageAccount.id, cpuCompute.id, storageBlobDataContributorRoleId)
  scope: storageAccount
  properties: {
    principalId: cpuCompute.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: storageBlobDataContributorRoleId
  }
}

output containerResourceId string = trainingContainer.id
output workspacePrincipalId string = workspace.identity.principalId
output computePrincipalId string = cpuCompute.identity.principalId