package train

import (
	"github.com/btracey/gofunopter/common"
	//"github.com/btracey/gofunopter/common/display"
	"github.com/btracey/nnet/loss"
	"github.com/btracey/nnet/nnet"
	"github.com/btracey/nnet/scale"

	"github.com/gonum/floats"
	//"math/rand"
	//"sync"

	//"fmt"
	"runtime"
)

// TrainAll trains on all of the input data. This is prone to overfitting,
// but may not be a problem if the input data is a good representation of
// the true underlying data. The input and output data are modified
// TODO: Should Inputs/Outputs really be public?
type TrainAll struct {
	net             *nnet.Net
	Inputs          [][]float64
	Outputs         [][]float64
	Weights         []float64
	chunkSize       int
	dLossDParam     [][][]float64
	dLossDParamFlat []float64
	nInputs         int
}

func NewTrainAll(net *nnet.Net, losser loss.Losser, inputs, outputs [][]float64, weights []float64) *TrainAll {
	t := &TrainAll{
		net:     net,
		Inputs:  inputs,
		Outputs: outputs,
		Weights: weights,
	}
	net.Losser = losser
	for _, weight := range weights {
		if weight < 0 {
			panic("negative weight")
		}
	}
	// Scale the weights
	sumWeights := floats.Sum(weights)
	floats.Scale(1/sumWeights, t.Weights)

	t.dLossDParam, t.dLossDParamFlat = net.NewPerParameterMemory()
	t.chunkSize = GetChunkSize(len(t.Inputs))
	return t
}

func (t *TrainAll) ObjGrad(parameters []float64) (loss float64, deriv []float64, err error) {
	t.net.SetParametersSlice(parameters)
	loss = nnet.ParLossDeriv(t.Inputs, t.Outputs, t.Weights, t.net, t.dLossDParam, t.chunkSize)

	// Don't need these here with weights
	//loss /= float64(len(t.Inputs))
	//floats.Scale(1/float64(len(t.Inputs)), t.dLossDParamFlat)
	return loss, t.dLossDParamFlat, nil
}

func (t *TrainAll) Scale() error {
	SetScale(t.Inputs, t.Outputs, t.net)
	err := scale.ScaleData(t.net.InputScaler, t.Inputs)
	if err != nil {
		return err
	}
	return scale.ScaleData(t.net.OutputScaler, t.Outputs)
}

func (t *TrainAll) Unscale() error {
	err := scale.UnscaleData(t.net.InputScaler, t.Inputs)
	if err != nil {
		return err
	}
	return scale.UnscaleData(t.net.OutputScaler, t.Outputs)
}

var TestLossIncrease common.Status = 100

// GetChunkSize returns the number of inputs per parallel goroutine
func GetChunkSize(inputs int) int {
	nCPU := runtime.GOMAXPROCS(-1)

	chunkSize := inputs / nCPU
	if chunkSize < 5 {
		chunkSize = 5
	}
	if chunkSize > 1000 {
		chunkSize = 1000
	}
	return chunkSize
}

func SetScale(inputs, outputs [][]float64, net *nnet.Net) error {
	err := net.InputScaler.SetScale(inputs)
	if err != nil {
		return err
	}
	return net.OutputScaler.SetScale(outputs)
}

func ScaleTrainingData(net *nnet.Net, trainInputs, trainOutputs, testInputs, testOutputs [][]float64) error {
	err := SetScale(trainInputs, trainOutputs, net)
	if err != nil {
		return err
	}
	err = scale.ScaleData(net.InputScaler, trainInputs)
	if err != nil {
		return err
	}

	err = scale.ScaleData(net.OutputScaler, trainOutputs)
	if err != nil {
		return err
	}
	if testInputs != nil {
		err = scale.ScaleData(net.InputScaler, testInputs)
		if err != nil {
			return err
		}
	}
	if testOutputs != nil {
		scale.ScaleData(net.OutputScaler, testOutputs)
		if err != nil {
			return err
		}
	}
	return nil
}

func UnscaleTrainingData(net *nnet.Net, trainInputs, trainOutputs, testInputs, testOutputs [][]float64) error {
	err := scale.UnscaleData(net.InputScaler, trainInputs)
	if err != nil {
		return err
	}

	err = scale.UnscaleData(net.OutputScaler, trainOutputs)
	if err != nil {
		return err
	}
	if testInputs != nil {
		err = scale.UnscaleData(net.InputScaler, testInputs)
		if err != nil {
			return err
		}
	}
	if testOutputs != nil {
		scale.UnscaleData(net.OutputScaler, testOutputs)
		if err != nil {
			return err
		}
	}
	return nil
}

/*
// OneFoldTrain trains the net by splitting the data into a training and a testing
// fold and stopping when the testing fold has
type OneFoldTrain struct {
	LossRatio float64

	net          *nnet.Net
	TrainInputs  [][]float64
	TrainOutputs [][]float64
	TestInputs   [][]float64
	TestOutputs  [][]float64
	chunkSize    int

	dLossDParamTrain     [][][]float64
	dLossDParamTrainFlat []float64
	dLossDParamTest      [][][]float64
	dLossDParamTestFlat  []float64
	trainLoss            float64
	testLoss             float64

	lossRatio float64

	KeepHistory bool // Keep the test loss history
	History     []float64
	nCalls      int
}

// NewOneFoldTrain does not copy data
func NewOneFoldTrain(net *nnet.Net, losser loss.Losser, inputs, outputs [][]float64) *OneFoldTrain {

	if len(inputs) != len(outputs) {
		panic("input and output lengths must match")
	}

	net.Losser = losser
	chunkSize := GetChunkSize(len(inputs))

	o := &OneFoldTrain{
		net:       net,
		chunkSize: chunkSize,
		LossRatio: 7,
	}

	// Divide the inputs and outputs in half
	nSamples := len(inputs)
	nTest := int(float64(nSamples) / 2)
	nTrain := nSamples - nTest
	rp := rand.Perm(nSamples)
	o.TestInputs = make([][]float64, nTest)
	o.TestOutputs = make([][]float64, nTest)
	o.TrainInputs = make([][]float64, nTrain)
	o.TrainOutputs = make([][]float64, nTrain)

	o.dLossDParamTrain, o.dLossDParamTrainFlat = net.NewPerParameterMemory()
	o.dLossDParamTest, o.dLossDParamTestFlat = net.NewPerParameterMemory()

	for i := range o.TestInputs {
		o.TestInputs[i] = inputs[rp[i]]
		o.TestOutputs[i] = outputs[rp[i]]
	}

	for i := range o.TrainInputs {
		o.TrainInputs[i] = inputs[rp[nTest+i]]
		o.TrainOutputs[i] = outputs[rp[nTest+i]]
	}
	return o
}

func (o *OneFoldTrain) Scale() error {
	err := ScaleTrainingData(o.net, o.TrainInputs, o.TrainOutputs, o.TestInputs, o.TestOutputs)
	if err != nil {
		return err
	}
	if o.net.Losser == nil {
		return fmt.Errorf("Net does not have Losser set")
	}
	return nil
}

func (o *OneFoldTrain) Unscale() error {
	return UnscaleTrainingData(o.net, o.TrainInputs, o.TrainOutputs, o.TestInputs, o.TestOutputs)
}

func (o *OneFoldTrain) Status() common.Status {
	if o.lossRatio > o.LossRatio {
		return TestLossIncrease
	}
	return common.Continue
}

func (o *OneFoldTrain) ObjGrad(weights []float64) (loss float64, deriv []float64, err error) {
	o.net.SetParametersSlice(weights)

	trainLossChan := make(chan float64)
	testLossChan := make(chan float64)
	go func() {
		trainLossChan <- nnet.ParLossDeriv(o.TrainInputs, o.TrainOutputs, o.net, o.dLossDParamTrain, o.chunkSize)
	}()
	go func() {
		testLossChan <- nnet.ParLossDeriv(o.TestInputs, o.TestOutputs, o.net, o.dLossDParamTest, o.chunkSize)
	}()

	w := sync.WaitGroup{}
	w.Add(2)
	go func() {
		o.trainLoss = <-trainLossChan
		o.trainLoss /= float64(len(o.TrainInputs))
		floats.Scale(1/float64(len(o.TrainInputs)), o.dLossDParamTrainFlat)
		w.Done()
	}()
	go func() {
		o.testLoss = <-testLossChan
		o.testLoss /= float64(len(o.TestInputs))

		floats.Scale(1/float64(len(o.TestInputs)), o.dLossDParamTestFlat)
		w.Done()
	}()
	w.Wait()

	o.lossRatio = o.testLoss / o.trainLoss

	o.AddToHistory(o.testLoss)
	return o.trainLoss, o.dLossDParamTrainFlat, nil
}

func (o *OneFoldTrain) AddToDisplay(d []*display.Struct) []*display.Struct {
	return append(d, &display.Struct{Heading: "TestLoss", Value: o.testLoss}, &display.Struct{Heading: "LossRatio", Value: o.lossRatio})
}

func (o *OneFoldTrain) AddToHistory(testLoss float64) {
	if o.KeepHistory {
		o.History = append(o.History, testLoss)
	}
}
*/
