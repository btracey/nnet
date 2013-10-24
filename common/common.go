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
	Type  reflect.Type
	Bytes []byte
}

type interfaceMarshaler struct {
	Type  reflect.Type
	Value interface{}
}

type typeUnmarshaler struct {
	Type reflect.Type
}

type valueUnmarshaler struct {
	Value interface{}
}

func (i *InterfaceMarshaler) MarshalJSON() ([]byte, error) {
	inter := interfaceMarshaler{
		Type:  reflect.TypeOf(i.Value),
		Value: i.Value,
	}
	return json.Marshal(inter)
}

type TypeMismatch struct {
	ValueType reflect.Type
	JSONType  reflect.Type
}

func (t TypeMismatch) Error() string {
	return fmt.Sprintf("Mismatch between provided type and JSON type. Provided type: %v, JSONType %v", t.ValueType, t.JSONType)
}

func (i *InterfaceMarshaler) UnmarshalJSON(data []byte) error {
	// Get the type
	t := &typeUnmarshaler{}
	i.Type = t.Type
	json.Unmarshal(data, t)
	if i.Value == nil {
		// Nothing more we can do
		i.Bytes = data
	}

	typeOfValue := reflect.TypeOf(i.Value)
	// If the value is not nil, check that the type matches
	if typeOfValue != t.Type {
		return TypeMismatch{ValueType: typeOfValue, JSONType: t.Type}
	}
	// If they do match, check if it's a pointer type
	isPtr := reflect.ValueOf(i.Value).Kind() == reflect.Ptr

	// If it is not a pointer, can't unmarshal a pointer to it.
	if !isPtr {
		return errors.New("Can't unmarshal a non-pointer type")
	}

	v := valueUnmarshaler{Value: i.Value}
	json.Unmarshal(data, v)
	i.Value = v.Value
	return nil
}
