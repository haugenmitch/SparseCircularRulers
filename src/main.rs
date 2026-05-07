use fixedbitset::FixedBitSet;

const TARGET_LENGTH: usize = 70;
const TARGET_SEGMENTS: usize = 10;

fn is_complete(segments: &[u8], total_length: usize) -> bool {
    // Create a bit set of ruler length - 1
    let mut measurable_lengths = FixedBitSet::with_capacity((total_length) as usize);

    // Add each possible length to the bit set
    for i in 0..segments.len() {
        let mut len: u16 = segments[i] as u16;
        measurable_lengths.insert(len as usize);
        for j in 1..(segments.len() - 1) {
            len += segments[(i + j) % segments.len()] as u16;
            measurable_lengths.insert(len as usize);
        }
    }

    // Check if every length has been set
    measurable_lengths.count_ones(..) == total_length - 1
}

fn find_rulers(current_segments: &mut [u8], current_index: usize, remaining_sum: usize) {
    // Remaining segments, not including the current segment
    let remaining_segments = TARGET_SEGMENTS - current_index - 1;

    if remaining_segments == 0 {
        current_segments[current_index] = remaining_sum as u8;
        if is_complete(current_segments, TARGET_LENGTH) {
            println!("{:?}", current_segments);
        }
        return;
    }

    // Leave at least 1 for each remaining segment (the last segment should be at least 2)
    // This is because there should be no trailing ones. Any ruler with trailing ones
    // could be shifted so the first trailing one becomes the first one of the ruler.
    let max_curr_segment_size = remaining_sum - (remaining_segments + 1);

    for i in 1..=max_curr_segment_size {
        current_segments[current_index] = i as u8;
        find_rulers(current_segments, current_index + 1, remaining_sum - i);
    }
}

fn main() {
    //println!("{:?}", is_complete(&[1, 2, 3, 4], TARGET_LENGTH));
    find_rulers(&mut [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1, TARGET_LENGTH - 1)
}
