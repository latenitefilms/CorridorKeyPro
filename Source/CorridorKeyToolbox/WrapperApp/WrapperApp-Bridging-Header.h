//
//  WrapperApp-Bridging-Header.h
//  CorridorKey by LateNite — Standalone Editor
//
//  Imports the C-side shader-types header so the wrapper app's Swift
//  code can reference the same `CKTextureIndex…`, `CKBufferIndex…`,
//  and parameter struct definitions the renderer plug-in uses. We
//  deliberately do NOT import the FxPlug SDK here — the wrapper
//  process never talks to FxPlug, and importing the SDK would force
//  the standalone build to depend on a framework only the renderer
//  plugin needs.
//

#ifndef CorridorKeyToolbox_WrapperApp_Bridging_Header_h
#define CorridorKeyToolbox_WrapperApp_Bridging_Header_h

#import "CorridorKeyShaderTypes.h"

#endif /* CorridorKeyToolbox_WrapperApp_Bridging_Header_h */
