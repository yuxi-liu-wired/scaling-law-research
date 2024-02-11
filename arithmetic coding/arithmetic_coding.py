class ArithmeticCode:
    def __init__(self, resolution=32):
        assert resolution == 32
        self.resolution = resolution
        self.high = 2**resolution - 1
        self.low = 0
        self.zero_bit_mask = 2 ** (resolution - 1)
        self.first_bit_mask = 2 ** (resolution - 2)
        self.emission_buffer = 0

    def decode(self, cdf, code):
        high, low, emission_buffer, resolution = (
            self.high,
            self.low,
            self.emission_buffer,
            self.resolution,
        )
        code_number = BitArray(resolution)
        code_number[0] = code[0]
        code_number[1:resolution] = code[
            emission_buffer + 1 : emission_buffer + resolution
        ]
        probability = (code_number[:resolution].uint32 - low) / (high - low)

        # Iterate through the CDF to find the symbol
        # The CDF is an ascending numpy array, so we can use binary search
        # `symbol` is defined by cdf[symbol] <= probability < cdf[symbol+1]
        symbol = np.searchsorted(cdf, probability, side="right") - 1
        assert cdf[symbol] <= probability < cdf[symbol + 1]
        interval = (cdf[symbol], cdf[symbol + 1])
        self.encode(interval)
        return symbol

    def encode(self, interval):
        cdf_low, cdf_high = interval
        high, low, emission_buffer, resolution = (
            self.high,
            self.low,
            self.emission_buffer,
            self.resolution,
        )
        code = BitArray(0)

        interval_length = (high + 1) - low

        # Adjust high and low based on the cdf intervals, convert to integer arithmetic
        high = low + int(interval_length * cdf_high) - 1
        low = low + int(interval_length * cdf_low)
        while (high & self.zero_bit_mask) == 0 or (low & self.zero_bit_mask) != 0:
            bitsegment = BitArray(emission_buffer + 1)
            bitsegment[0] = 1
            if (high & self.zero_bit_mask) == 0:
                code.append(~bitsegment)
            else:
                code.append(bitsegment)
            high <<= 1  # Shift left and add 1 as the least significant bit
            high += 1
            low <<= 1
            high %= 2**self.resolution
            low %= 2**self.resolution
            emission_buffer = 0
        # Adjust for close ranges
        while (high & self.first_bit_mask == 0) and (low & self.first_bit_mask != 0):
            emission_buffer += 1
            high <<= 1
            high |= self.zero_bit_mask + 1
            high %= 2**self.resolution
            low <<= 1
            low %= self.zero_bit_mask
        self.high, self.low, self.emission_buffer = high, low, emission_buffer
        return code
