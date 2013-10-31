package common

import (
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
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

func InterfaceFullTypename(i interface{}) string {
	pkgpath, pkgname := InterfaceLocation(i)
	return filepath.Join(pkgpath, pkgname)
}

func InterfaceLocation(i interface{}) (pkgpath string, name string) {
	if reflect.ValueOf(i).Kind() == reflect.Ptr {
		pkgpath = reflect.ValueOf(i).Elem().Type().PkgPath()
		name = reflect.ValueOf(i).Elem().Type().Name()
	} else {
		pkgpath = reflect.TypeOf(i).PkgPath()
		name = reflect.TypeOf(i).Name()
	}
	return
}

func (i *InterfaceMarshaler) MarshalJSON() ([]byte, error) {
	// Need to get the type of the value, not the pointer to the value
	inter := interfaceMarshaler{
		Value: i.Value,
	}

	inter.PkgPath, inter.Name = InterfaceLocation(i.Value)
	b, err := json.Marshal(inter)
	if err != nil {
		fmt.Println("common In error")
		return b, fmt.Errorf("nnet/common: error marshaling interface: " + err.Error())
	}
	return b, nil
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

var NoValue = errors.New("nnet/common: no value provided for unmarshaling")

func (i *InterfaceMarshaler) UnmarshalJSON(data []byte) error {
	// Get the type
	t := &typeUnmarshaler{}
	err := json.Unmarshal(data, t)
	if err != nil {
		return err
	}
	i.PkgPath = t.PkgPath
	i.Name = t.Name
	if i.Value == nil {
		// Nothing more we can do
		i.Bytes = data
		return NoValue
	}

	typeOfValue := reflect.ValueOf(i.Value).Elem().Type()

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
