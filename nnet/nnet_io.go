package nnet

import (
	"github.com/btracey/nnet/common"
	"github.com/btracey/nnet/loss"
	"github.com/btracey/nnet/scale"

	"bytes"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
)

func (net *Net) GobEncode() ([]byte, error) {
	w := new(bytes.Buffer)
	var err error
	encoder := gob.NewEncoder(w)
	err = encoder.Encode(&net.Losser)
	if err != nil {
		return nil, fmt.Errorf("Error encoding Losser: %v", err)
	}

	err = encoder.Encode(&net.InputScaler)
	if err != nil {
		return nil, fmt.Errorf("Error encoding input scaler: %v", err)
	}

	err = encoder.Encode(&net.OutputScaler)
	if err != nil {
		return nil, fmt.Errorf("Error encoding output scaler: %v", err)
	}

	err = encoder.Encode(net.nInputs)
	if err != nil {
		return nil, fmt.Errorf("Error encoding nInputs: %v", err)
	}

	err = encoder.Encode(net.layers)
	if err != nil {
		return nil, err
	}

	err = encoder.Encode(&net.parameters)
	if err != nil {
		return nil, err
	}
	return w.Bytes(), nil
}

type netMarshal struct {
	Losser                 *common.InterfaceMarshaler
	InputScaler            *common.InterfaceMarshaler
	OutputScaler           *common.InterfaceMarshaler
	NumInputs              int
	NumOutputs             int
	TotalNumParameters     int
	NumParametersPerNeuron [][]int
	ParameterIndex         [][]int
	Parameters             []float64
	NumLayers              int
	NumNeuronsPerLayer     []int
	Layers                 []Layer
	PredictionCheck        []predictionCheck
}

// predictionCheck provides a checksum that the net loaded in properly
type predictionCheck struct {
	Input  []float64
	Output []float64
}

const nPredictionCheck = 5

// Save the net to a string file (for reading in to non-go programs for example). If custom
// interfaces are used, they will be marshaled with custom and if the custom type is a text
// marshaller it will write
func (net *Net) MarshalJSON() (b []byte, err error) {
	nLayers := len(net.nParameters)
	nNeuronsPerLayer := make([]int, nLayers)
	for i := range nNeuronsPerLayer {
		nNeuronsPerLayer[i] = len(net.nParameters[i])
	}

	predChecks := make([]predictionCheck, nPredictionCheck)
	inputs := make([][]float64, nPredictionCheck)
	for i := range inputs {
		inputs[i] = make([]float64, net.nInputs)
		for j := range inputs[i] {
			inputs[i][j] = rand.Float64()
		}
	}

	outputs, err := net.PredictSlice(inputs)
	if err != nil {
		return nil, err
	}

	for i := range predChecks {
		predChecks[i].Input = inputs[i]
		predChecks[i].Output = outputs[i]
	}

	n := &netMarshal{

		Losser:                 &common.InterfaceMarshaler{I: net.Losser},
		InputScaler:            &common.InterfaceMarshaler{I: net.InputScaler},
		OutputScaler:           &common.InterfaceMarshaler{I: net.OutputScaler},
		NumInputs:              net.nInputs,
		NumOutputs:             net.nOutputs,
		TotalNumParameters:     net.totalNumParameters,
		NumLayers:              nLayers,
		NumNeuronsPerLayer:     nNeuronsPerLayer,
		NumParametersPerNeuron: net.nParameters,
		ParameterIndex:         net.parameterIdx,
		Parameters:             net.parametersSlice,
		Layers:                 net.layers,
		PredictionCheck:        predChecks,
	}
	return json.Marshal(n)
}

type lossUnmarshaler struct {
	Losser *common.InterfaceMarshaler
}

// UnmarshalJSON unmarshals the net. If the interfaces are not part
// of the nnet suite, they must be
func (net *Net) UnmarshalJSON(data []byte) error {
	v := &netMarshal{}
	err := json.Unmarshal(data, v)
	if err != nil {
		return fmt.Errorf("nnet/net/unmarshaljson: error unmarshaling data: " + err.Error())
	}
	// Now, unpack all the values
	net.Losser = v.Losser.I.(loss.Losser)
	net.InputScaler = v.InputScaler.I.(scale.Scaler)
	net.OutputScaler = v.OutputScaler.I.(scale.Scaler)
	net.nInputs = v.NumInputs
	net.nOutputs = v.NumOutputs
	net.totalNumParameters = v.TotalNumParameters
	// TODO: Add layers
	net.nParameters = v.NumParametersPerNeuron
	net.parameterIdx = v.ParameterIndex
	//net.parametersSlice = v.Parameters
	net.layers = v.Layers

	net.parameters, net.parametersSlice = net.NewPerParameterMemory()
	for i, val := range v.Parameters {
		net.parametersSlice[i] = val
	}

	return nil
}

// GobDecode some comment about needing to register custom types
func (net *Net) GobDecode(buf []byte) error {
	r := bytes.NewBuffer(buf)
	decoder := gob.NewDecoder(r)

	var err error

	err = decoder.Decode(&net.Losser)
	if err != nil {
		return fmt.Errorf("Error decoding losser: %v", err)
	}

	err = decoder.Decode(&net.InputScaler)
	if err != nil {
		return fmt.Errorf("Error decoding input scaler: %v", err)
	}

	err = decoder.Decode(&net.OutputScaler)
	if err != nil {
		return fmt.Errorf("Error decoding output scaler: %v", err)
	}
	err = decoder.Decode(&net.nInputs)
	if err != nil {
		return fmt.Errorf("Error decoding nInputs: %v", err)
	}

	err = decoder.Decode(&net.layers)
	if err != nil {
		return fmt.Errorf("Error decoding layers: %v", err)
	}
	net.new()
	err = decoder.Decode(&net.parameters)
	if err != nil {
		return fmt.Errorf("Error decoding parameters: %v", err)
	}

	return nil
}

// Save saves the neural net
func (net *Net) Save(filename string) error {
	bytes, err := net.GobEncode()
	if err != nil {
		return err
	}
	return ioutil.WriteFile(filename, bytes, 0700)
}

// Load loads in a neural net from a file.
func Load(filename string) (*Net, error) {
	bytes, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	net := &Net{}
	err = net.GobDecode(bytes)
	if err != nil {
		return nil, err
	}
	return net, nil
}
