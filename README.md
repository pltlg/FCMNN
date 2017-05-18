# FCMNN

This library aims to provide a simplistic implementation of a fully connected and multilayered neural network in Swift language.

---


## How to add?

Just drag FCMNN.swift into your project. Simple as that.

A demo project also lives in this repo.

___


## Basic terms

The goal of the neural network is to solve problems in the same way that a human would, although several neural network categories are more abstract. 

**Feedforward**  
A feedforward neural network is an artificial neural network wherein connections between the units do not form a cycle. As such, it is different from recurrent neural networks. The feedforward neural network was the first and simplest type of artificial neural network devised. In this network, the information moves in only one direction, forward, from the input nodes, through the hidden nodes (if any) and to the output nodes. There are no cycles or loops in the network.

**Multi-layer**
This class of networks consists of multiple layers of computational units, usually interconnected in a feed-forward way. Each neuron in one layer has directed connections to the neurons of the subsequent layer. In many applications the units of these networks apply a sigmoid function as an activation function.

**Activation**
The activation function that converts a neuron's weighted input to its output activation.
![Activation](https://cloud.githubusercontent.com/assets/5107640/26205146/2d526f6a-3be1-11e7-86d6-5021e2da55a4.png)  

**Fully connected layers**
 The high-level reasoning in the neural network is done via fully connected layers. Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks. Their activations can hence be computed with a matrix multiplication followed by a bias offset.

**Bias**  
Biases are values that are added to the sums calculated at each node (except input nodes) during the feedforward. For simplicity, biases are commonly visualized simply as values associated with each node in the intermediate and output layers of a network, but in practice are treated in exactly the same manner as other weights, with all biases simply being weights associated with vectors that lead from a single node whose location is outside of the main network and whose activation is always 1.

![Bias](https://cloud.githubusercontent.com/assets/5107640/26205216/624eca60-3be1-11e7-893d-541d917b9147.png)  

**Weights**
The synapses store parameters called "weights" that manipulate the data in the calculations. The weights of the interconnections, which are updated in the learning process. The FCMNN is built for the feedforward phase, so you have to import a weight set for the configured model. You can add your weights according to the following order:

![Weights](https://cloud.githubusercontent.com/assets/5107640/26205227/683e6ad4-3be1-11e7-98b9-48c5e8f228a9.png)  

___


## How to configure?. <a name="use" id="use"></a>

```swift
//1. number of the layers
static var numberOfLayers = 4

//2. number of neurons within the input layer
static var inputNeurons = 4

//3. number of the neurons within the hidden layers
static var hiddenNeurons = [10,10]

//4. number of the output neurons within the output layer
static var outputNeurons = 2

//5. weights
static var weights : [[Double]] = [

//INPUT -> 1.HIDDEN
[-1.121302230,-17.23721300,1.177202312,1.711212210,-1.733312711,-1.212711713,-7.113333332,-1.120277722,1.001271377,1.273737322,-1.212172013,2.207172771,-1.733111002,1.173137713,-2.173070372,1.221131111,1.201011122,1.331701313,-1.221721223,2.373310732,1.711130771,3.717707272,-1.720322221,-1.1371771,1.773721127,-1.777272717,1.230327077,-1.720331102,-1.337177211,3.032711103,-1.311321221,-2.2017112,-1.207203713,-1.132272323,-1.127011202,1.232237213,3.772212071,-1.202231012,2.231313120,1.712271123,-1.110171170,233.2703122,3.332772111,-17.12710111,7.003271237,1.103772202,-2.213720123,1.722131311,-1.210371323,-1.217721211],

//1.HIDDEN -> 2.HIDDEN
[1.011122122,1.212103737,23.17732732,-2.731313711,-1.112233123,1.123011212,1.3171313,-1.173113732,-0.711177121,-1.710771117,1.221173272,-1.317221707,1.123101137,-2.337273337,-1.230127372,2.337711130,1.032202177,-1.221217371,-1.177011731,1.127211010,-72.33330711,-1.133713723,-1.110213710,-0.171217013,7.202731177,2.023210271,3.773213101,-7.732073317,-31.13737171,-2.321173131,1.33773722,3.271211232,2.330111303,2.123377132,-1.72731031,-7.771117312,-1.130232131,23.31333301,-31.31221133,-271.1770311,1.122230130,-2.201220033,737.3312220,-1.331713113,1.113307331,-2.327231773,7.371222322,1.37137720,3.713337072,1.31217122,1.723712711,-1.773717211,-1.131203132,-2.727710307,-1.173120732,1.711111211,0.727271711,-1.11103127,1.330131232,-1.737110322,1.122227712,3.137013770,1.102273117,1.111073332,1.212231112,-7.013303073,-1.21077127,3.722112730,1.31731772,-1.277037322,2.301011172,1.230311777,3.322322712,1.101113112,1.712301033,-1.122171112,1.117711022,1.212221227,-1.711213021,13.12727120,2.772131111,1.313177771,1.133233317,-1.277370120,1.130772372,-2.711373710,1.337711172,-2.207130330,-1.710001311,1.177112307,-23.17133000,3.217137032,1.11271120,3.110131377,1.211117233,1.171713717,1-1.32111217,-2.172721101,-2.177132131,1.732171310,72.12127122,-7.133723317,3.072711170,1.301701112,7.231373713,-7.123131333,3.132133271,-37.70302171,-7.232122272,-17.17773101],

//2.HIDDEN -> OUTPUT
[-1.701717112,322.7177311,-1.7211170,133.3371117,-1.321312,-12.13377237,-230.7117173,1172.172332,3727.011312,-1.3371703,1117.722021,1.720312377,-327.323221,-770.7270011,-137.1312720,277.1133107,12.71111313,237.1312010,-1133.717031,-3773.010237,-370.1211312,-1103.20311]
]

//6. network object
let network = FCMNN(inputs: inputNeurons, layers: numberOfLayers, hiddens: hiddenNeurons, outputs: outputNeurons, weights: weights)

//7. input
let nnInput : [Double] = [ 4.2, 5.2, 1.2, 5.5 ]

//8. recieve the result
let output: [Double] = try network.fire(inputs: nnInput)


```

