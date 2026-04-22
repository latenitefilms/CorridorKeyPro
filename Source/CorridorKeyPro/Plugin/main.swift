//
//  main.swift
//  Corridor Key Pro
//
//  Entry point for the FxPlug XPC service. `FxPrincipal.startServicePrincipal`
//  hands control over to Final Cut Pro's plug-in host, which instantiates
//  `CorridorKeyProPlugIn` on demand. No other setup is required here.
//

import Foundation

FxPrincipal.startServicePrincipal()
