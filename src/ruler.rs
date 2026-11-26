//! Ruler facts used throughout: the canonical form and the segment-count lower bound.

/// Whether `ruler` (segments, starting with a 1) is the lexicographically smallest of its
/// rotations and reflections. Only those that start at a segment of 1 can be smaller.
pub fn is_canonical(ruler: &[u8]) -> bool {
    let n = ruler.len();
    for i in (0..n).filter(|&i| ruler[i] == 1) {
        let rotation_smaller = i > 0 && is_smaller(ruler, |j| ruler[(i + j) % n]);
        let reflection_smaller = is_smaller(ruler, |j| ruler[(i + n - j) % n]);
        if rotation_smaller || reflection_smaller {
            return false;
        }
    }
    true
}

/// Whether the sequence `other(0), other(1), ...` is lexicographically smaller than `ruler`.
fn is_smaller(ruler: &[u8], other: impl Fn(usize) -> u8) -> bool {
    for (j, &a) in ruler.iter().enumerate() {
        let b = other(j);
        if b != a {
            return b < a;
        }
    }
    false
}

/// Fewest segments a complete ruler of `length` can have (see ALGORITHM.md, section 1).
pub fn num_segments_lower_bound(length: u8) -> u8 {
    let n = (((length as f64 * 4.0 - 3.0).sqrt() + 1.0) / 2.0).ceil() as u8;
    // A ruler meeting the bound exactly is a perfect difference set of order n - 1, which
    // exists only for prime-power orders.
    let order = (n as u32).saturating_sub(1);
    if order > 1 && length as u32 == order * order + order + 1 && !is_prime_power(order) {
        n + 1
    } else {
        n
    }
}

/// Whether `q` is a power of a prime.
fn is_prime_power(mut q: u32) -> bool {
    if q < 2 {
        return false;
    }
    let p = (2..=q).find(|p| q % p == 0).unwrap();
    while q % p == 0 {
        q /= p;
    }
    q == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_canonical() {
        assert!(is_canonical(&[1, 1, 3, 6]));
        // A reflection and a rotation of [1, 1, 3, 6].
        assert!(!is_canonical(&[1, 1, 6, 3]));
        assert!(!is_canonical(&[1, 3, 6, 1]));

        // Symmetric rulers.
        assert!(is_canonical(&[1, 1, 2, 2]));
        assert!(!is_canonical(&[1, 2, 2, 1]));
        assert!(is_canonical(&[1, 2, 1, 2]));

        // Several 1s.
        assert!(is_canonical(&[1, 1, 2, 1, 1, 3]));
        assert!(!is_canonical(&[1, 1, 3, 1, 1, 2]));
    }

    #[test]
    fn test_num_segments_lower_bound() {
        assert_eq!(num_segments_lower_bound(6), 3);
        assert_eq!(num_segments_lower_bound(13), 4);
        assert_eq!(num_segments_lower_bound(31), 6);
        // Perfect difference sets: orders 6, 10, and 12 are not prime powers; 7 and 8 are.
        assert_eq!(num_segments_lower_bound(43), 8);
        assert_eq!(num_segments_lower_bound(111), 12);
        assert_eq!(num_segments_lower_bound(157), 14);
        assert_eq!(num_segments_lower_bound(57), 8);
        assert_eq!(num_segments_lower_bound(73), 9);
        assert_eq!(num_segments_lower_bound(44), 8);
        assert_eq!(num_segments_lower_bound(3), 2);
        assert_eq!(num_segments_lower_bound(21), 5);
    }
}
