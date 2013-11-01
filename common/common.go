package common

import (
	"encoding/json"
	"errors"
	"path/filepath"
	"reflect"

	//"fmt"
)

func init() {
	interfaceMap = make(map[string]interface{})
	//	isPtr = make(map[string]bool)
}

// interfaceMap helps encoding and decoding interfaces
var interfaceMap map[string]interface{}

//var isPtr map[string]bool

func RegisterString(i interface{}) string {
	str := InterfaceFullTypename(i)
	// See if it's a pointer, and if so append a * at the end
	if reflect.ValueOf(i).Kind() == reflect.Ptr {
		str += "*"
	}

	return str
}

// Register panics if the type is already registered (like Gob)
func Register(i interface{}) {
	str := RegisterString(i)
	_, ok := interfaceMap[str]
	if ok {
		panic("nnet/common interface type already registered")
	}

	isPtr := reflect.ValueOf(i).Kind() == reflect.Ptr

	var newVal interface{}
	var tmp interface{}
	if isPtr {
		tmp = reflect.ValueOf(i).Elem().Interface()
	} else {
		tmp = i
	}
	newVal = reflect.New(reflect.TypeOf(tmp)).Elem().Interface()

	// Either way, save a real value

	//TODO: Add in something where the types aren't copied for the *
	interfaceMap[str] = newVal

	//interfaceMap[str] = i
}

var NotRegistered error = errors.New("nnet/loss: losser type not registered")

func ptrFromString(str string) (interface{}, bool, error) {
	val, ok := interfaceMap[str]
	if !ok {
		return nil, false, NotRegistered
	}
	isPtr := str[len(str)-1:len(str)] == "*"

	return reflect.New(reflect.TypeOf(val)).Interface(), isPtr, nil
}

// FromString returns a copy of the losser
func FromString(str string) (interface{}, error) {
	val, isPtr, err := ptrFromString(str)
	if err != nil {
		return nil, err
	}
	/*
		// Make a copy of that type, and then dereference it
		var newVal interface{}
		if isPtr {
			newVal = reflect.New(reflect.TypeOf(val)).Interface()
		} else {
			newVal = reflect.New(reflect.TypeOf(val)).Elem().Interface()
		}
	*/
	var newVal interface{}
	newVal = val
	if !isPtr {
		newVal = reflect.ValueOf(val).Elem().Interface()
	}
	return newVal, nil
}

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

type InterfaceMarshaler struct {
	I interface{}
}

type lossMarshaler struct {
	Type  string
	Value interface{}
}

type typeUnmarshaler struct {
	Type  string
	Value json.RawMessage
}

func (l *InterfaceMarshaler) MarshalJSON() ([]byte, error) {
	name := RegisterString(l.I)
	// Check that the type has been registered
	_, ok := interfaceMap[name]
	if !ok {
		return nil, NotRegistered
	}
	marshaler := &lossMarshaler{
		Type:  name,
		Value: l.I,
	}
	return json.Marshal(marshaler)
}
func (l *InterfaceMarshaler) UnmarshalJSON(data []byte) error {
	// First, unmarshal the type
	t := &typeUnmarshaler{}
	err := json.Unmarshal(data, t)
	if err != nil {
		return err
	}
	// Get a pointer to the right type
	val, isPtr, err := ptrFromString(t.Type)
	if err != nil {
		return errors.New("nnet/common error unmarshaling: " + err.Error())
	}

	nextdata := []byte(t.Value)
	// Unmarshal the (pointer to the) value
	err = json.Unmarshal(nextdata, val)
	if err != nil {
		return err
	}

	// If we don't want an interface, return a pointer to it
	if !isPtr {
		val = reflect.ValueOf(val).Elem().Interface()
	}

	l.I = val
	return nil
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

/*
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
*/
