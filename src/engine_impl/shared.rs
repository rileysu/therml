use crate::{engine::{tensor::{builder::EngineTensorBuilder, factory::EngineTensorFactory, EngineTensor}, unit::UnitCompatible}, helper::{position, Interval, Position, Shape, Slice, Stride, VarArrayCompatible}};

use super::tensor::padded::Padded;

//a: (batches, in_channels, img_y, img_x)
//kernel_shape: (in_channels, k_y, k_x)
//out: (batches, in_channels, out_y, out_x, k_y * k_x)
pub fn im2col_2d<T: UnitCompatible, E: EngineTensorFactory<Unit = T>>(
    a: &dyn EngineTensor<Unit = T>,
    kernel_shape: &Shape,
    padding: usize,
    stride: usize,
) -> Box<dyn EngineTensor<Unit = T>> {
    let batches = a.shape().get(0).unwrap();
    let in_channels = a.shape().get(1).unwrap();

    //Ok if zero padding
    let a_padded = Padded::pad_from(
        a.clone(),
        [0, 0, padding, padding].as_slice().into(),
        T::zero(),
    );

    let img_y = a_padded.shape().get(2).unwrap();
    let img_x = a_padded.shape().get(3).unwrap();

    let k_y = kernel_shape.get(1).unwrap();
    let k_x = kernel_shape.get(2).unwrap();

    let out_y = (img_y - k_y) / stride + 1;
    let out_x = (img_x - k_x) / stride + 1;

    let patch_len = k_y * k_x;

    let out_shape = Shape::from([batches, in_channels, out_y, out_x, patch_len].as_slice());
    //let out_stride = Stride::default_from_shape(&out_shape);

    //let final_img_dims = Shape::new(kernel_shape.iter().zip(img_dims.iter()).map(|(k_d, a_d)| (a_d + 2 * padding - k_d) / stride + 1).collect());

    let grouped_patches_shape = Shape::from([batches, in_channels, patch_len].as_slice());

    //Buffer used for output

    //let mut buffer = Vec::<T>::from_iter(iter::repeat(T::zero()).take(out_shape.elements()));
    //buffer.shrink_to_fit();

    let mut builder = E::builder(out_shape.clone(), T::default());

    for y in 0..out_y {
        for x in 0..out_x {
            let grouped_patches = a_padded.slice(
                [
                    Interval::all(),
                    Interval::all(),
                    Interval::between_with_step(y, y + k_y, stride),
                    Interval::between_with_step(x, x + k_x, stride),
                ]
                .as_slice(),
            );

            let grouped_patches = grouped_patches.reshape(&grouped_patches_shape);

            for batch in 0..batches {
                for channel in 0..in_channels {
                    let patch = grouped_patches.slice(
                        &[
                            Interval::only(batch),
                            Interval::only(channel),
                            Interval::all(),
                        ],
                    );

                    //let start_index = Position::from([batch, channel, y, x, 0].as_slice())
                    //    .tensor_index(&out_stride)
                    //    .unwrap();

                    //buffer.splice(start_index..(start_index + patch_len), patch.iter_units());

                    let start_pos = position![batch, channel, y, x, 0];
                    let last_pos = position![batch, channel, y, x, patch_len - 1];
                    builder.splice_between_positions(&start_pos, &last_pos, patch.iter_units());
                }
            }
        }
    }

    //E::from_slice(buffer.as_slice(), out_shape).generic()

    builder.construct().generic()
}

#[cfg(test)]
mod test {
    use crate::engine_impl::tensor::array::Array;

    use super::*;

    #[test]
    fn simple_im2col_2d() {
        //Pytorch generated im2col
        let expected: [f32; 972] = [
            0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 4.0, 5.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 1.0, 2.0, 0.0, 4.0, 5.0, 0.0,
            7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0,
            8.0, 9.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            9.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 11.0, 0.0, 13.0, 14.0, 0.0, 0.0, 0.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 0.0,
            0.0, 0.0, 11.0, 12.0, 0.0, 14.0, 15.0, 0.0, 0.0, 10.0, 11.0, 0.0, 13.0, 14.0, 0.0,
            16.0, 17.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 11.0, 12.0, 0.0,
            14.0, 15.0, 0.0, 17.0, 18.0, 0.0, 0.0, 13.0, 14.0, 0.0, 16.0, 17.0, 0.0, 0.0, 0.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 0.0, 0.0, 0.0, 14.0, 15.0, 0.0, 17.0, 18.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 19.0, 20.0, 0.0, 22.0, 23.0, 0.0, 0.0, 0.0, 19.0,
            20.0, 21.0, 22.0, 23.0, 24.0, 0.0, 0.0, 0.0, 20.0, 21.0, 0.0, 23.0, 24.0, 0.0, 0.0,
            19.0, 20.0, 0.0, 22.0, 23.0, 0.0, 25.0, 26.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
            26.0, 27.0, 20.0, 21.0, 0.0, 23.0, 24.0, 0.0, 26.0, 27.0, 0.0, 0.0, 22.0, 23.0, 0.0,
            25.0, 26.0, 0.0, 0.0, 0.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 0.0, 0.0, 0.0, 23.0,
            24.0, 0.0, 26.0, 27.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 28.0, 29.0, 0.0, 31.0,
            32.0, 0.0, 0.0, 0.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 0.0, 0.0, 0.0, 29.0, 30.0,
            0.0, 32.0, 33.0, 0.0, 0.0, 28.0, 29.0, 0.0, 31.0, 32.0, 0.0, 34.0, 35.0, 28.0, 29.0,
            30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 29.0, 30.0, 0.0, 32.0, 33.0, 0.0, 35.0, 36.0,
            0.0, 0.0, 31.0, 32.0, 0.0, 34.0, 35.0, 0.0, 0.0, 0.0, 31.0, 32.0, 33.0, 34.0, 35.0,
            36.0, 0.0, 0.0, 0.0, 32.0, 33.0, 0.0, 35.0, 36.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 37.0, 38.0, 0.0, 40.0, 41.0, 0.0, 0.0, 0.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0,
            0.0, 0.0, 0.0, 38.0, 39.0, 0.0, 41.0, 42.0, 0.0, 0.0, 37.0, 38.0, 0.0, 40.0, 41.0, 0.0,
            43.0, 44.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 38.0, 39.0, 0.0,
            41.0, 42.0, 0.0, 44.0, 45.0, 0.0, 0.0, 40.0, 41.0, 0.0, 43.0, 44.0, 0.0, 0.0, 0.0,
            40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 0.0, 0.0, 0.0, 41.0, 42.0, 0.0, 44.0, 45.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 46.0, 47.0, 0.0, 49.0, 50.0, 0.0, 0.0, 0.0, 46.0,
            47.0, 48.0, 49.0, 50.0, 51.0, 0.0, 0.0, 0.0, 47.0, 48.0, 0.0, 50.0, 51.0, 0.0, 0.0,
            46.0, 47.0, 0.0, 49.0, 50.0, 0.0, 52.0, 53.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0,
            53.0, 54.0, 47.0, 48.0, 0.0, 50.0, 51.0, 0.0, 53.0, 54.0, 0.0, 0.0, 49.0, 50.0, 0.0,
            52.0, 53.0, 0.0, 0.0, 0.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 0.0, 0.0, 0.0, 50.0,
            51.0, 0.0, 53.0, 54.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 55.0, 56.0, 0.0, 58.0,
            59.0, 0.0, 0.0, 0.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 0.0, 0.0, 0.0, 56.0, 57.0,
            0.0, 59.0, 60.0, 0.0, 0.0, 55.0, 56.0, 0.0, 58.0, 59.0, 0.0, 61.0, 62.0, 55.0, 56.0,
            57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 56.0, 57.0, 0.0, 59.0, 60.0, 0.0, 62.0, 63.0,
            0.0, 0.0, 58.0, 59.0, 0.0, 61.0, 62.0, 0.0, 0.0, 0.0, 58.0, 59.0, 60.0, 61.0, 62.0,
            63.0, 0.0, 0.0, 0.0, 59.0, 60.0, 0.0, 62.0, 63.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 64.0, 65.0, 0.0, 67.0, 68.0, 0.0, 0.0, 0.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0,
            0.0, 0.0, 0.0, 65.0, 66.0, 0.0, 68.0, 69.0, 0.0, 0.0, 64.0, 65.0, 0.0, 67.0, 68.0, 0.0,
            70.0, 71.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 65.0, 66.0, 0.0,
            68.0, 69.0, 0.0, 71.0, 72.0, 0.0, 0.0, 67.0, 68.0, 0.0, 70.0, 71.0, 0.0, 0.0, 0.0,
            67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 0.0, 0.0, 0.0, 68.0, 69.0, 0.0, 71.0, 72.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 73.0, 74.0, 0.0, 76.0, 77.0, 0.0, 0.0, 0.0, 73.0,
            74.0, 75.0, 76.0, 77.0, 78.0, 0.0, 0.0, 0.0, 74.0, 75.0, 0.0, 77.0, 78.0, 0.0, 0.0,
            73.0, 74.0, 0.0, 76.0, 77.0, 0.0, 79.0, 80.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
            80.0, 81.0, 74.0, 75.0, 0.0, 77.0, 78.0, 0.0, 80.0, 81.0, 0.0, 0.0, 76.0, 77.0, 0.0,
            79.0, 80.0, 0.0, 0.0, 0.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 0.0, 0.0, 0.0, 77.0,
            78.0, 0.0, 80.0, 81.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 82.0, 83.0, 0.0, 85.0,
            86.0, 0.0, 0.0, 0.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 0.0, 0.0, 0.0, 83.0, 84.0,
            0.0, 86.0, 87.0, 0.0, 0.0, 82.0, 83.0, 0.0, 85.0, 86.0, 0.0, 88.0, 89.0, 82.0, 83.0,
            84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 83.0, 84.0, 0.0, 86.0, 87.0, 0.0, 89.0, 90.0,
            0.0, 0.0, 85.0, 86.0, 0.0, 88.0, 89.0, 0.0, 0.0, 0.0, 85.0, 86.0, 87.0, 88.0, 89.0,
            90.0, 0.0, 0.0, 0.0, 86.0, 87.0, 0.0, 89.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 91.0, 92.0, 0.0, 94.0, 95.0, 0.0, 0.0, 0.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0,
            0.0, 0.0, 0.0, 92.0, 93.0, 0.0, 95.0, 96.0, 0.0, 0.0, 91.0, 92.0, 0.0, 94.0, 95.0, 0.0,
            97.0, 98.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 92.0, 93.0, 0.0,
            95.0, 96.0, 0.0, 98.0, 99.0, 0.0, 0.0, 94.0, 95.0, 0.0, 97.0, 98.0, 0.0, 0.0, 0.0,
            94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 0.0, 0.0, 0.0, 95.0, 96.0, 0.0, 98.0, 99.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 101.0, 0.0, 103.0, 104.0, 0.0, 0.0, 0.0,
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 0.0, 0.0, 0.0, 101.0, 102.0, 0.0, 104.0,
            105.0, 0.0, 0.0, 100.0, 101.0, 0.0, 103.0, 104.0, 0.0, 106.0, 107.0, 100.0, 101.0,
            102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 101.0, 102.0, 0.0, 104.0, 105.0, 0.0,
            107.0, 108.0, 0.0, 0.0, 103.0, 104.0, 0.0, 106.0, 107.0, 0.0, 0.0, 0.0, 103.0, 104.0,
            105.0, 106.0, 107.0, 108.0, 0.0, 0.0, 0.0, 104.0, 105.0, 0.0, 107.0, 108.0, 0.0, 0.0,
            0.0, 0.0,
        ];

        let batches = 4_usize;
        let in_channels = 1_usize;
        let y = 3_usize;
        let x = 3_usize;

        let k_y = 3_usize;
        let k_x = 3_usize;

        let a_shape = Shape::from([batches, in_channels, y, x].as_slice());
        let kernel_shape = Shape::from([in_channels, k_y, k_x].as_slice());

        let a = Array::from_iter(
            (1..=(batches * in_channels * y * x)).map(|x| x as f32),
            a_shape,
        );

        let res = im2col_2d::<_, Array<_>>(a.generic().as_ref(), &kernel_shape, 1, 1);

        for (res_element, expected_element) in res.iter_units().zip(expected.iter()) {
            assert_eq!(res_element, *expected_element);
        }
    }
}
