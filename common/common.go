package common

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
)

// NotInPackage is an error which signifies the type is
// not in the package. Normally used with marshaling and
// unmarshaling
var NotInPackage = errors.New("NotInPackage")

// UnmarshalMismatch is an error type used when unmarshaling the specific
// activators in this package.
type UnmarshalMismatch struct {
	Expected string
	Received string
}

// Error is so UnmarshalMismatch implements the error interface
func (u UnmarshalMismatch) Error() string {
	return "Unmarshal string mismatch. Expected: " + u.Expected + " Received: " + u.Received
}

// InterfaceMarshaler is a helper struct for marshaling and unmarshaling interfaces
// When marshaling, only the Value field is useful
// When unmarshaling, the Type field will always be marshaled. If there is
// a value in the marshaler
type InterfaceMarshaler struct {
	Value interface{}
	Type  string
	Bytes []byte
}

type interfaceMarshaler struct {
	Type  string
	Value interface{}
}

type typeUnmarshaler struct {
	Type string
}

type valueUnmarshaler struct {
	Value interface{}
}

func (i *InterfaceMarshaler) MarshalJSON() ([]byte, error) {
	inter := interfaceMarshaler{
		Type:  reflect.TypeOf(i.Value).String(),
		Value: i.Value,
	}
	return json.Marshal(inter)
}

type TypeMismatch struct {
	ValueType string
	JSONType  string
}

func (t TypeMismatch) Error() string {
	return fmt.Sprintf("Mismatch between provided type and JSON type. Provided type: %v, JSONType %v", t.ValueType, t.JSONType)
}

var NoValue = errors.New("No value provided for unmarshaling")

func (i *InterfaceMarshaler) UnmarshalJSON(data []byte) error {
	// Get the type
	t := &typeUnmarshaler{}
	i.Type = t.Type
	err := json.Unmarshal(data, t)
	if err != nil {
		return err
	}
	if i.Value == nil {
		// Nothing more we can do
		i.Bytes = data
		return NoValue
	}

	typeOfValue := reflect.TypeOf(i.Value).String()
	// If the value is not nil, check that the type matches
	if typeOfValue != t.Type {
		return TypeMismatch{ValueType: typeOfValue, JSONType: t.Type}
	}
	// Unmarshal the value: Will return an error if the value is not a pointer
	v := &valueUnmarshaler{Value: i.Value}
	err = json.Unmarshal(data, v)
	if err != nil {
		return err
	}
	i.Value = v.Value
	return nil
}
