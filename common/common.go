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
	Value   interface{}
	PkgPath string
	Name    string
	Bytes   []byte
}

type interfaceMarshaler struct {
	PkgPath string
	Name    string
	Value   interface{}
}

type typeUnmarshaler struct {
	PkgPath string
	Name    string
}

type valueUnmarshaler struct {
	Value interface{}
}

func (i *InterfaceMarshaler) MarshalJSON() ([]byte, error) {
	fmt.Println(i.Value)
	inter := interfaceMarshaler{
		PkgPath: reflect.TypeOf(i.Value).Elem().PkgPath(),
		Name:    reflect.TypeOf(i.Value).Elem().Name(),
		Value:   i.Value,
	}
	fmt.Printf("inter: %#v", inter)
	return json.Marshal(inter)
}

type TypeMismatch struct {
	ValuePkg  string
	JSONPkg   string
	ValueName string
	JSONName  string
}

func (t TypeMismatch) Error() string {
	if t.ValueName == "" {
		fmt.Sprintf("nnet/common: package mismatch. Provided pkg: %v, JSON pkg %v", t.ValuePkg, t.JSONPkg)
	}
	return fmt.Sprintf("nnet/common: name mismatch. Provided name: %v, JSON name %v", t.ValueName, t.JSONName)
}

var NoValue = errors.New("No value provided for unmarshaling")

func (i *InterfaceMarshaler) UnmarshalJSON(data []byte) error {
	// Get the type
	t := &typeUnmarshaler{}
	i.PkgPath = t.PkgPath
	i.Name = t.Name
	err := json.Unmarshal(data, t)
	if err != nil {
		return err
	}
	if i.Value == nil {
		// Nothing more we can do
		i.Bytes = data
		return NoValue
	}

	typeOfValue := reflect.TypeOf(i.Value)

	// If the value is not nil, check that the type matches
	if typeOfValue.PkgPath() != t.PkgPath {
		return TypeMismatch{
			ValuePkg: typeOfValue.PkgPath(),
			JSONPkg:  t.PkgPath,
		}
	}
	if typeOfValue.Name() != t.Name {
		return TypeMismatch{
			ValueName: typeOfValue.Name(),
			JSONName:  t.Name,
		}
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
