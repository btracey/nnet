// Write random predictions to a file so they can be read by other nnet code
// to verify consistency
// Need to figure out oddness with registering types
//TODO: Should make this a function, not a main package

package main

import (
	"encoding/json"
	"fmt"
	"github.com/btracey/nnet/nnet"
	"math/rand"
	"os"
	"path/filepath"
)

func main() {

	if len(os.Args) != 3 {
		fmt.Println("Must have two arguments: netFilename and writeFilename")
	}

	netFilename := os.Args[1]
	writeFilename := os.Args[2]

	// make sure that the net filename is a .gob
	ext := filepath.Ext(netFilename)
	if ext != ".gob" {
		fmt.Println("First argument must be a .gob file")
		os.Exit(1)
	}

	// Make sure the other filename is a .txt
	ext = filepath.Ext(writeFilename)
	if ext != ".txt" {
		fmt.Println("Second argument must be a .txt file")
		os.Exit(2)
	}

	//
	net, err := nnet.Load(netFilename)
	if err != nil {
		fmt.Println("Error loading .gob file: " + err.Error())
	}

	// Generate some random points to predict
	nPoints := 10
	nInputs := net.Inputs()

	pts := make([][]float64, nPoints)
	for i := range pts {
		pts[i] = make([]float64, nInputs)
		for j := range pts[i] {
			pts[i][j] = rand.Float64()
		}
	}

	// Unscale the inputs (to put them in the proper range)
	for i := range pts {
		net.InputScaler.Unscale(pts[i])
	}

	// Predict at all the points
	outputs, err := net.PredictSlice(pts)
	if err != nil {
		fmt.Println("Error predicting: " + err.Error())
		os.Exit(1)
	}

	// Write the ouputs to a file
	f, err := os.Create(writeFilename)
	defer f.Close()
	if err != nil {
		fmt.Println("Error opening write file: " + err.Error())
		os.Exit(1)
	}

	list := make([]struct {
		Inputs  []float64
		Outputs []float64
	}, len(pts))
	for i := range pts {
		list[i].Inputs = pts[i]
		list[i].Outputs = outputs[i]
	}
	b, err := json.MarshalIndent(list, "", "\t")
	if err != nil {
		fmt.Println("Error marshaling outputs: " + err.Error())
		os.Exit(1)
	}
	_, err = f.Write(b)
	if err != nil {
		fmt.Println("Error writing to file: " + err.Error())
		os.Exit(1)
	}
}
