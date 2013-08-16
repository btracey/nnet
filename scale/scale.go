package scale

import (
	//"fmt"
	"math"
)

// Finds the appropriate scaling of the data such that the dataset has a mean of 0 and a variance of 1.
// If the true standard deviation of any of the inputs is zero (for example because there is only one input, or if all of that input is constant), the standard deviation is set to 1.0 to avoid NaNs
func FindScale(input [][]float64) (mean, std []float64) {
	// Need to find the mean input and the std of the input
	nDim := len(input[0])
	mean = make([]float64, nDim)
	for _, samp := range input {
		for i, val := range samp {
			mean[i] += val
		}
	}
	for i := range mean {
		mean[i] /= float64(len(input))
	}
	std = make([]float64, nDim)
	for _, samp := range input {
		for i, val := range samp {
			std[i] += math.Pow((val - mean[i]), 2)
		}
	}
	for i := range std {
		std[i] /= float64(len(input))
		std[i] = math.Sqrt(std[i])
	}
	for i := range std {
		if std[i] == 0 {
			std[i] = 1.0
		}
	}
	return mean, std
}

func ScaleData(points [][]float64, mean, std []float64) {
	for _, val := range points {
		ScalePoint(val, mean, std)
	}
}

func ScalePoint(point []float64, mean, std []float64) {
	for i := range point {
		point[i] = (point[i] - mean[i]) / std[i]
	}
}

func UnscaleData(points [][]float64, mean, std []float64) {
	for _, val := range points {
		UnscalePoint(val, mean, std)
	}
}

func UnscalePoint(point []float64, mean, std []float64) {
	for i := range point {
		point[i] = point[i]*std[i] + mean[i]
	}
}
