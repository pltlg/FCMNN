//
//  FCMNN,swift
//
//  Created by gaborpletl on 2/6/17.
//

import Accelerate
import Foundation

public final class FCMNN {
    var iNeurNum : Int, hLayerNum : Int, hNeurNum : [Int], oNeurNum : Int, weights : [[Double]]
    var iCache : [Double], oCache : [Double], lCache : [[Double]]

    /**
     Initializes a the neural network.
     */
    init(inputs: Int, layers: Int, hiddens: [Int], outputs: Int, weights: [[Double]]) {

        self.iNeurNum = inputs
        self.hLayerNum = layers - 2
        self.hNeurNum = hiddens
        self.oNeurNum = outputs
        self.weights = weights

        self.iCache = Array(repeating: 0, count: self.iNeurNum+1)
        self.lCache = Array()
        self.lCache.append(Array(repeating: 0, count: self.iNeurNum+1))
        for layerCount in hiddens {
            self.lCache.append(Array(repeating: 0, count: layerCount+1))
        }
        self.oCache = Array(repeating: 0, count: self.oNeurNum)
    }

    /**
     The fire(inputs:[Double])->[Double] function runs the neural network with the given `inputs` double array parameter.
     */
    public func fire(inputs: [Double]) -> [Double] {
        self.iCache[0] = 1.0
        for i in 1...self.iNeurNum {
            self.iCache[i] = inputs[i - 1]
        }

        lCache[0] = iCache
        for layer in 1...self.hLayerNum {
            let currWeight = self.weights[layer-1]
            var currCache = self.lCache[layer]
            let prevCache = self.lCache[layer-1]

            vDSP_mmulD(currWeight, 1,
                       prevCache, 1,
                       &currCache, 1,
                       vDSP_Length(currCache.count-1), vDSP_Length(1), vDSP_Length(prevCache.count))

            self.lCache[layer] =  mask(array: currCache.shift(by: 1), till: 1, value: 1)

            self.activate(layer: layer)

        }

        vDSP_mmulD(self.weights.last!, 1,
                   self.lCache.last!, 1,
                   &self.oCache, 1,
                   vDSP_Length(self.oNeurNum), vDSP_Length(1), vDSP_Length(self.hNeurNum.last!+1))

        activateOutput()
        let result = self.oCache

        return result
    }

    /**
     Activate the given layer.
     */
    private func activate(layer : Int) {
        for i in (1...self.hNeurNum[layer-1]).reversed() {
            self.lCache[layer][i] = sigmoid(x: self.lCache[layer][i])
        }
        self.lCache[layer][0] = 1.0
    }
    /**
     Activate the output layers.
     */
    private func activateOutput() {
        for i in 0..<self.oNeurNum {
            self.oCache[i] = sigmoid(x: self.oCache[i])
        }
    }

    private func sigmoid(x: Double) -> Double {
        return 1 / (1 + exp(-x))
    }

    private func mask(array : [Double], till : Int, value : Double) -> [Double] {
        var rArray = array
        for i in 0..<till {
            rArray[i] = value
        }
        return rArray
    }
}

extension Array {
    // 12345 -> 45123
    // This does not accept a negative amount
    func shift(by amount : Int) -> Array {
        let amountFromEnd = amount % self.count
        return Array(self.suffix(amountFromEnd) + self.prefix(self.count - amountFromEnd))
    }
}
