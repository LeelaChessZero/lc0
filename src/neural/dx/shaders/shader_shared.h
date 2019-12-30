#define kExpandPlanesElementsPerBlock 256
#define kExpandPlanesFp32BlockSize kExpandPlanesElementsPerBlock
#define kExpandPlanesFp16BlockSize (kExpandPlanesElementsPerBlock / 2)

// for both input transform and output transform shaders
#define kWinogradTransformShaderBlockSize 64

#define kConv1x1BlockSize 64

#define kAddVectorsBlockSize 512