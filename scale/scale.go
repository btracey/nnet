package scale

import (
	"errors"
	"github.com/gonum/floats"
	"math"

	//"fmt"
)

// IdenticalDimensions is an error type expressing that
// a dimension all had equal values. Dims is a list of unequal dimensions
type UniformDimension struct {
	Dims []int
}

func (i *UniformDimension) Error() string {
	return "Some dimensions had all values with the same entry"
}

type UnequalLength struct{}

func (u UnequalLength) Error() string {
	return "Data length mismatch"
}

// Scalar is an interface for transforming data so it is appropriately scaled
// for the machine learning algorithm. The data are a slice of data points.
// All of the data points must have equal lengths. An error is returned if
// some of the data have unequal lengths or if less than two data points are
// entered
type Scaler interface {
	Scale(point []float64) error     // Scales (in place) the data point
	Unscale(point []float64) error   // Unscales (in place) the data point
	IsScaled() bool                  // Returns true if the scale for this type has already been set
	Dimensions() int                 //Number of dimensions for wihich the data was scaled
	SetScale(data [][]float64) error // Uses the input data to set the scale
}

// ScaleData scales every point in the data using the scaler
func ScaleData(scaler Scaler, data [][]float64) error {
	if len(data) == 0 {
		return nil
	}
	for _, point := range data {
		err := scaler.Scale(point)
		if err != nil {
			return err
		}
	}
	return nil
}

// UnscaleData scales every point in the data using the scalar
func UnscaleData(scaler Scaler, data [][]float64) error {
	if len(data) == 0 {
		return nil
	}
	for _, point := range data {
		err := scaler.Unscale(point)
		if err != nil {
			return err
		}
	}
	return nil
}

// checkInputs checks the inputs to make sure they all have
// equal length and that there are at least two inputs
func checkInputs(data [][]float64) error {
	if !floats.EqualLengths(data...) {
		return UnequalLength{}
	}
	if len(data) < 2 {
		return errors.New("LessThanTwoInputs")
	}
	return nil
}

// None is a type specifying no transformation of the input should be done
type None struct {
	Dim    int // Dimensions
	Scaled bool
}

func (n None) IsScaled() bool {
	return n.Scaled
}

func (n None) Scale(x []float64) error {
	return nil
}

func (n None) Unscale(x []float64) error {
	return nil
}

func (n None) Dimensions() int {
	return n.Dim
}

func (n *None) SetScale(data [][]float64) error {
	err := checkInputs(data)
	if err != nil {
		return err
	}
	n.Dim = len(data[0])
	n.Scaled = true
	return nil
}

// Linear is a type for scaling the data to be between 0 and 1
type Linear struct {
	Min    []float64 // Maximum value of the data
	Max    []float64 // Minimum value of the data
	Scaled bool      // Flag if the scale has been set
	Dim    int       // Number of dimensions of the data
}

// IsScaled returns true if the scale has been set
func (l *Linear) IsScaled() bool {
	return l.Scaled
}

// Dimensions returns the length of the data point
func (l *Linear) Dimensions() int {
	return l.Dim
}

// SetScale sets a linear scale between 0 and 1. If no data
// points. If the minimum and maximum value are identical in
// a dimension, the minimum and maximum values will be set to
// that value +/- 0.5 and a
func (l *Linear) SetScale(data [][]float64) error {

	err := checkInputs(data)
	if err != nil {
		return err
	}

	dim := len(data[0])

	// Generate data for min and max if they don't already exist
	if len(l.Min) < dim {
		l.Min = make([]float64, dim)
	} else {
		l.Min = l.Min[0:dim]
	}
	if len(l.Max) < dim {
		l.Max = make([]float64, dim)
	} else {
		l.Max = l.Max[0:dim]
	}
	for i := range l.Min {
		l.Min[i] = math.Inf(1)
	}
	for i := range l.Max {
		l.Max[i] = math.Inf(-1)
	}
	// Find the minimum and maximum in each dimension
	for _, point := range data {
		for i, val := range point {
			if val < l.Min[i] {
				l.Min[i] = val
			}
			if val > l.Max[i] {
				l.Max[i] = val
			}
		}
	}
	l.Scaled = true
	l.Dim = dim

	var unifError *UniformDimension

	// Check that the maximum and minimum values are not identical
	for i := range l.Min {
		if l.Min[i] == l.Max[i] {
			if unifError == nil {
				unifError = &UniformDimension{}
			}
			unifError.Dims = append(unifError.Dims, i)
			l.Min[i] -= 0.5
			l.Max[i] += 0.5
		}
	}
	if unifError != nil {
		return unifError
	}
	return nil
}

// Scales the point returning an error if the length doesn't match
func (l *Linear) Scale(point []float64) error {
	if len(point) != l.Dim {
		return UnequalLength{}
	}
	for i, val := range point {
		point[i] = (val - l.Min[i]) / (l.Max[i] - l.Min[i])
	}
	return nil
}

func (l *Linear) Unscale(point []float64) error {
	if len(point) != l.Dim {
		return UnequalLength{}
	}
	for i, val := range point {
		point[i] = val*(l.Max[i]-l.Min[i]) + l.Min[i]
	}
	return nil
}

// Normal scales the data to have a mean of 0 and a variance of 1
// in each dimension
type Normal struct {
	Mu     []float64
	Sigma  []float64
	Dim    int
	Scaled bool
}

// IsScaled returns true if the scale has been set
func (n *Normal) IsScaled() bool {
	return n.Scaled
}

// Dimensions returns the length of the data point
func (n *Normal) Dimensions() int {
	return n.Dim
}

// Finds the appropriate scaling of the data such that the dataset has
//  a mean of 0 and a variance of 1. If the standard deviation of any of
// the data is zero (all of the entries have the same value),
// the standard deviation is set to 1.0 and a UniformDimension error is
// returned
func (n *Normal) SetScale(data [][]float64) error {

	err := checkInputs(data)
	if err != nil {
		return err
	}

	// Need to find the mean input and the std of the input
	dim := len(data[0])
	mean := make([]float64, dim)
	for _, samp := range data {
		for i, val := range samp {
			mean[i] += val
		}
	}
	for i := range mean {
		mean[i] /= float64(len(data))
	}

	std := make([]float64, dim)
	for _, samp := range data {
		for i, val := range samp {
			diff := val - mean[i]
			std[i] += diff * diff
		}
	}
	for i := range std {
		std[i] /= float64(len(data))
		std[i] = math.Sqrt(std[i])
	}
	n.Scaled = true
	n.Dim = dim

	var unifError *UniformDimension
	for i := range std {
		if std[i] == 0 {
			if unifError == nil {
				unifError = &UniformDimension{}
			}
			unifError.Dims = append(unifError.Dims, i)
			std[i] = 1.0
		}
	}

	n.Mu = mean
	n.Sigma = std
	if unifError != nil {
		return unifError
	}
	return nil
}

// Scale scales the data point
func (n *Normal) Scale(point []float64) error {
	if len(point) != n.Dim {
		return UnequalLength{}
	}
	for i := range point {
		point[i] = (point[i] - n.Mu[i]) / n.Sigma[i]
	}
	return nil
}

// Unscale unscales the data point
func (n *Normal) Unscale(point []float64) error {
	if len(point) != n.Dim {
		return UnequalLength{}
	}
	for i := range point {
		point[i] = point[i]*n.Sigma[i] + n.Mu[i]
	}
	return nil
}

type ProbabilityDistribution interface {
	Fit([]float64) error
	CumProb(float64) float64
	Quantile(float64) float64
	Prob(float64) float64
}

// Probability scales the inputs based on the supplied
// probability distributions
type Probability struct {
	UnscaledDistribution []ProbabilityDistribution // Probabilitiy distribution from which the data come
	ScaledDistribution   []ProbabilityDistribution // Probability distribution to which the data should be scaled
	Dim                  int
	Scaled               bool
}

// IsScaled returns true if the scale has been set
func (p *Probability) IsScaled() bool {
	return p.Scaled
}

// Dimensions returns the length of the data point
func (p *Probability) Dimensions() int {
	return p.Dim
}

func (p *Probability) SetScale(data [][]float64) error {
	err := checkInputs(data)
	if err != nil {
		return err
	}
	p.Dim = len(data[0])
	if len(p.UnscaledDistribution) != p.Dim {
		return errors.New("Number of unscaled probability distributions must equal dimension")
	}
	if len(p.ScaledDistribution) != p.Dim {
		return errors.New("Unscaled distribution not set")
	}

	tmp := make([]float64, len(data))
	for i := 0; i < p.Dim; i++ {
		// Collect all the data into tmp
		for j, point := range data {
			tmp[j] = point[i]
		}
		// Fit the probability distribution using the samples
		p.UnscaledDistribution[i].Fit(tmp)
	}
	return nil
}

func (p *Probability) Scale(point []float64) error {
	if len(point) != p.Dim {
		return UnequalLength{}
	}
	for i := range point {
		// Check that the point doesn't have zero probability
		if p.UnscaledDistribution[i].Prob(point[i]) == 0 {
			return errors.New("Zero probability point")
		}
		//fmt.Println("point", point[i])
		prob := p.UnscaledDistribution[i].CumProb(point[i])
		//fmt.Println("point i", point[i], "prob", prob)
		//fmt.Println("prob", prob)
		point[i] = p.ScaledDistribution[i].Quantile(prob)
		//fmt.Println("newpoint", point[i])
		if math.IsInf(point[i], 0) {
			panic("inf point")
		}
		if math.IsNaN(point[i]) {
			panic("NaN point")
		}
	}
	return nil
}

func (p *Probability) Unscale(point []float64) error {
	if len(point) != p.Dim {
		return UnequalLength{}
	}
	for i := range point {
		// Check that the point doesn't have zero probability
		if p.UnscaledDistribution[i].Prob(point[i]) == 0 {
			return errors.New("Zero probability point")
		}
		prob := p.ScaledDistribution[i].CumProb(point[i])
		point[i] = p.UnscaledDistribution[i].Quantile(prob)
	}
	return nil
}
