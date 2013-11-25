package nnet

import (
	"encoding/json"
	"github.com/btracey/nnet/common"
)

// layer represents a layer of neurons
type Layer struct {
	Neurons []Neuron
}

func (l Layer) MarshalJSON() ([]byte, error) {
	n := make([]*common.InterfaceMarshaler, len(l.Neurons))
	for i := range l.Neurons {
		n[i] = &common.InterfaceMarshaler{I: l.Neurons[i]}
	}
	return json.Marshal(n)
}

func (l *Layer) UnmarshalJSON(data []byte) error {
	v := make([]*common.InterfaceMarshaler, 0)
	err := json.Unmarshal(data, &v)
	if err != nil {
		return err
	}
	l.Neurons = make([]Neuron, len(v))
	for i := range l.Neurons {
		l.Neurons[i] = v[i].I.(Neuron)
	}
	return nil
}
