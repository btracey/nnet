package common

import (
	//pkgbytes "bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"
)

type Foo struct {
	Dave float64
}

type Bar struct {
	Baz int
}

type Int int

type Quux struct {
}

func tryRegister(i interface{}) (b bool) {
	defer func() {
		if r := recover(); r != nil {
			b = true
		}
	}()
	Register(i)
	return
}

func TestRegisterAndFromString(t *testing.T) {
	// Clear the map
	interfaceMap = make(map[string]interface{})
	// Registering a type should work
	Register(Bar{})
	Register(&Foo{})
	// Also registering a pointer to that type should work
	Register(&Bar{})
	i := Int(0)
	Register(i)

	if !tryRegister(Bar{}) {
		t.Errorf("Should panic when registering the same type twice")
	}

	// Try reading from the types
	// non-pointer type
	t1 := Bar{}
	val, err := FromString(RegisterString(t1))
	if err != nil {
		t.Errorf("Error in from string: " + err.Error())
	}
	if !reflect.DeepEqual(val, t1) {
		t.Errorf("non-pointer doesn't match. Found: %v Expected: %v", reflect.TypeOf(val), reflect.TypeOf(t1))
	}
	// pointer type
	t2 := &Bar{}
	val2, err := FromString(RegisterString(t2))
	if err != nil {
		t.Errorf("Error in from string: " + err.Error())
	}
	// Values of pointers shouldn't be equal, but the types of pointers should be equal
	if !reflect.DeepEqual(val2, t2) {
		fmt.Println(val2)
		fmt.Println(t2)
		fmt.Printf("%p\n", val2)
		fmt.Printf("%p\n", t2)

		t.Errorf("pointer doesn't match. Found: %v. Expected: %v", reflect.TypeOf(val2), reflect.TypeOf(t2))
	}

	// TODO: Test to confirm they aren't teh same pointer

	/*
		if !reflect.DeepEqual(reflect.TypeOf(val2), reflect.TypeOf(t2)) {
			t.Errorf("pointer doesn't match. Found: %v. Expected: %v", reflect.TypeOf(val2), reflect.TypeOf(t2))
		}
	*/

	// Try the integer type
	t3 := Int(0)
	val3, err := FromString(RegisterString(t1))
	if err != nil {
		t.Errorf("Error in from string: " + err.Error())
	}
	if !reflect.DeepEqual(val, t1) {
		t.Errorf("non-pointer doesn't match. Found: %v Expected: %v", reflect.TypeOf(val3), reflect.TypeOf(t3))
	}

	// Try a type that doesn't exist
	_, err = FromString(RegisterString(Quux{}))
	if err != NotRegistered {
		t.Errorf("should return error on unregistered type")
	}
}

func TestMarshal(t *testing.T) {
	interfaceMap = make(map[string]interface{})
	Register(Bar{})
	Register(&Bar{})
	i := Int(0)
	Register(i)
	var err error

	err = testMarshal(Bar{6})
	if err != nil {
		t.Errorf("non-pointer struct: " + err.Error())
	}

	err = testMarshal(&Bar{6})
	if err != nil {
		t.Errorf("pointer struct: " + err.Error())
	}

	err = testMarshal(Int(15))
	if err != nil {
		t.Errorf("integer value: " + err.Error())
	}

}

var DoesNotMatch error = errors.New("doesn't match")

func testMarshal(i interface{}) error {
	v := &InterfaceMarshaler{I: i}
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("error marshaling: " + err.Error())
	}

	v2 := &InterfaceMarshaler{}
	err = json.Unmarshal(b, v2)
	if err != nil {
		return fmt.Errorf("error unmarshaling" + err.Error())
	}

	// If v is a non-pointer type, compare the actual values
	//	isPtr := reflect.ValueOf(v).Kind() == reflect.Ptr
	//	if !isPtr {
	//		fmt.Println("not pointer")
	if !reflect.DeepEqual(v.I, v2.I) {
		fmt.Println("doesn't match")
		fmt.Printf("%#v\n", v.I)
		fmt.Printf("%#v\n", v2.I)
		return DoesNotMatch
	}
	return nil
	//	}
	/*
		// Otherwise, compare the values teh pointers reference
		if !reflect.DeepEqual(reflect.ValueOf(v).Elem().Interface(), reflect.ValueOf(v2).Elem().Interface()) {
			fmt.Println(v)
			fmt.Println(reflect.TypeOf(v))
			fmt.Println(v2)
			fmt.Println(reflect.TypeOf(v2))
			return DoesNotMatch
		}
	*/
	return nil
}

/*
func TestInterfaceMarshaler(t *testing.T) {
	pointer := &Pointer{A: 10, B: "More Filler"}
	iface := Foo(pointer)

	// Test marshaling
	marshaler := &InterfaceMarshaler{Value: iface}
	bytes, err := json.Marshal(marshaler)
	fmt.Println("marshaled pointer bytes: ", string(bytes))
	if err != nil {
		t.Errorf("Error marshaling")
	}
	unmarshaler := &InterfaceMarshaler{}
	err = json.Unmarshal(bytes, unmarshaler)
	if err != NoValue {
		if err == nil {
			t.Errorf("No error returned when value not provided")
		}
		t.Errorf("Error unmarshaling: " + err.Error())
	}
	unmarshaler.Value = Foo(&Pointer{})
	err = json.Unmarshal(bytes, unmarshaler)
	if err != nil {
		t.Errorf("Error unmarshalling with value: " + err.Error())
	}

	// Check that the deep equal works
	if !reflect.DeepEqual(pointer, unmarshaler.Value) {
		t.Errorf("Not equal after marshal and unmarshal")
		fmt.Println(pointer)
		fmt.Println(unmarshaler.Value)
	}

	// Try marshaling a non-pointer type
	nonpointer := Nonpointer{C: 20, D: "Few"}
	iface2 := Foo(nonpointer)
	marshaler = &InterfaceMarshaler{Value: iface2}

	bytes, err = json.Marshal(marshaler)
	fmt.Println("marshaled non-pointer bytes: ", string(bytes))
	if err != nil {
		t.Errorf("Error marshaling non-pointer struct: " + err.Error())
	}
	// Should be able to unmarshal a pointer to that type
	np2 := &Nonpointer{}
	iface3 := Foo(np2)
	unmarshaler = &InterfaceMarshaler{Value: iface3}
	err = json.Unmarshal(bytes, unmarshaler)
	// Check that the underlying values are the same
	if !reflect.DeepEqual(nonpointer, reflect.ValueOf(unmarshaler.Value).Elem().Interface()) {
		t.Errorf("Unmarshaled nonpointer doesn't match")
	}
	unmarshaler.Value = Pointer{}
	err = json.Unmarshal(bytes, unmarshaler)
	if err == nil {
		t.Errorf("Should be an error for non-pointer value")
	}

	// Marshaling a struct and a pointer to a struct should be the same
	one := Nonpointer{C: 20, D: "Few"}
	two := &Nonpointer{C: 20, D: "Few"}
	bytes, err = json.Marshal(one)
	bytes2, err := json.Marshal(two)
	if !pkgbytes.Equal(bytes, bytes2) {
		t.Errorf("Marshaling not the same when marshaling struct and pointer to struct")
	}
}
*/
