package main

import (
	//"bytes"
	"encoding/json"
	"fmt"
	"github.com/btracey/nnet/nnet"
	"os"
	"path"
)

func main() {
	// Check that there is an argument
	if len(os.Args) < 2 {
		fmt.Println("No arguments provided. Must provide the .gob file")
		os.Exit(1)
	}
	if len(os.Args) > 3 {
		fmt.Println("More than two arguments provided")
	}
	// The filename is provided as the first argument
	filename := os.Args[1]

	// Check that the file is a .gob file
	ext := path.Ext(filename)
	if ext != ".gob" {
		fmt.Println("Provided file not a .gob file")
		os.Exit(2)
	}

	// Load the gob file
	net, err := nnet.Load(filename)
	if err != nil {
		fmt.Println("Error loading net from file: " + err.Error())
		os.Exit(3)
	}

	// JSON the net
	b, err := json.MarshalIndent(net, "", "\t")
	if err != nil {
		fmt.Println("Error marshaling JSON: " + err.Error())
	}

	/*
		buf := make([]byte, len(b)+100)
		dst := bytes.NewBuffer(buf)
		err = json.Indent(dst, b, "", "\t")
		if err != nil {
			fmt.Println("Err indenting json")
			os.Exit(4)
		}
	*/
	var jsonname string
	if len(os.Args) == 3 {
		jsonname = os.Args[2]
	} else {
		// Save the net as a JSON file with the same name but a JSON extension
		jsonname = filename[:len(filename)-len(ext)] + ".json"
	}
	f, err := os.Create(jsonname)
	defer f.Close()
	if err != nil {
		fmt.Println("Error creating new file")
		os.Exit(5)
	}
	f.Write(b)

}
