# Operations

Check indictates support

* Pointwise Single
    * [x] abs
    * [x] neg

    * [ ] average_pool
    * [ ] batch_norm

    * [ ] relu
    * [ ] leakyrelu
    * [ ] sigmoid
    
* Pointwise Scalar (broadcast?)
    * [x] add_scalar
    * [x] sub_scalar_lh (s - t)
    * [x] sub_scalar_rh (t - s)
    * [x] mul_scalar
    * [x] div_scalar_lh (s / t)
    * [x] div_scalar_rh (t / s)

* Pointwise Double
    * [x] add
    * [x] sub
    * [x] mul
    * [x] div

* Reduction
    * [ ] max
    * [ ] min 
    * [ ] sum
    * [ ] product

* Creation
    * [ ] zeroes
    * [ ] ones

* Utility
    * [x] shape
    * [x] stride
    * [x] iter
    * [x] slice
    * [x] reshape
    * [ ] concat

* Not Supported (Out of scope operations or units that aren't supported)
    * [ ] affine_grid
    * [ ] and
    * [ ] arg_max
    * [ ] arg_min
