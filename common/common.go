package common

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
