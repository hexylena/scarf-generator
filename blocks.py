import random
import tqdm
import sys
import copy
import svg


def divisors(num):
    i = 1
    while i < num:
        if num % i == 0:
            yield i
        i = i + 1


def avg(data):
    return sum(data) / len(data)


def empty_space(data):
    it = list(iterator(data))
    out = list(gpm(data, it, '/'))
    return avg(out)


def invert(data):
    for r in range(len(data)):
        row = []
        for c in range(len(data[r])):
            row.append(not data[r][c])
        yield row


def iterator(data):
    for r in range(len(data)):
        for c in range(len(data[r])):
            yield (c, r)

def symmetry_horiz(data):
    good = []

    # rounding down misses the middle spot
    m = len(data[0]) // 2

    for r in range(len(data)):
        for i in range(m):
            good.append(data[r][i] == data[r][len(data[r]) - 1 - i])

    return avg(good)



def trim(data, width=None, height=None):
    data = list(data)
    for r in range(len(data)):
        if height:
            if height > 0:
                if r >= height:
                    continue
            else:
                if r - len(data) < height:
                    continue

        if width:
            yield data[r][:width]
        else:
            yield data[r]


def subtract(a, b):
    for r in range(len(a)):
        row = []
        for c in range(len(a[r])):
            row.append(a[r][c] and not b[r][c])
        yield row


def overlay(a, b):
    for r in range(len(a)):
        row = []
        for c in range(len(a[r])):
            row.append(a[r][c] or b[r][c])
        yield row

def xor(a, b):
    for r in range(len(a)):
        row = []
        for c in range(len(a[r])):
            row.append(a[r][c] ^ b[r][c])
        yield row


def intersect(a, b):
    for r in range(len(a)):
        row = []
        for c in range(len(a[r])):
            row.append(a[r][c] and b[r][c])
        yield row


def gs(data, x, y, value, bounds=True):
    if x < 0 or y < 0:
        return

    try:
        data[x][y] = value
    except IndexError:
        pass

def eat(data, distance, diag=False):
    for i in range(distance):
        data = list(_eat(data, diag=diag))

    return data

def _eat(data, diag=False):
    for r in range(len(data)):
        row = []
        for c in range(len(data[r])):
            if data[r][c]:
                row.append(True)
            else:
                q = list(bb([[True]], x=c, y=r, corners=diag))
                row.append(any(gpm(data, q)))
        yield row


def gpm(data, positions, other=False):
    for (x, y) in positions:
        yield gp(data, x, y, other=other)


def gp(data, x, y, other=False):
    if x < 0 or y < 0:
        return other

    try:
        return data[y][x]
    except IndexError:
        return other

def printerr(*args):
    sys.stderr.write(str(args) + "\n")


def _empty(height, width, value=False):
    for y in range(height):
        yield [value] * width

def empty(*args):
    return list(_empty(*args))

def bb(shape, x, y, corners=True):
    if corners:
        # top edge
        for i in range(len(shape[0]) + 2):
            yield (x + i - 1, y - 1)
        # bottom edge
        for i in range(len(shape[0]) + 2):
            yield (x + i - 1, y + len(shape[0]))

        # left edge
        for i in range(len(shape) + 2):
            yield (x - 1, y - 1 + i)

        # right edge
        for i in range(len(shape) + 2):
            yield (x + len(shape), y - 1 + i)
    else:
        # top edge
        for i in range(len(shape[0])):
            yield (x + i, y - 1)
        # bottom edge
        for i in range(len(shape[0])):
            yield (x + i, y + len(shape[0]))

        # left edge
        for i in range(len(shape)):
            yield (x - 1, y + i)

        # right edge
        for i in range(len(shape)):
            yield (x + len(shape), y + i)


def blit(data, shape, x=0, y=0, gap=False, throw=False):
    data = list(data)
    shape = list(shape)
    bounding_box = list(bb(shape, x, y))

    for r in range(len(shape)):
        for c in range(len(shape[0])):
            if shape[r][c]:
                if not gap:
                    try:
                        data[y + r][x + c] = shape[r][c]
                    except Exception as e:
                        if throw:
                            raise e
                else:
                    # take mask, eat, & operation, if nay.
                    if not any([gp(data, x, y, False) for (x, y) in bounding_box]):
                        try:
                            data[y + r][x + c] = shape[r][c]
                        except Exception as e:
                            if throw:
                                raise e


def mirror_horizontal(shape):
    """
    V
    A
    """
    data = []
    for r in range(len(shape)):
        row = []
        for c in range(len(shape[0])):
            row.append(shape[len(shape) - r - 1][c])
        data.append(row)
    return data


def mirror_vertical(shape):
    """
    SZ
    """
    data = []
    for r in range(len(shape)):
        row = []
        for c in range(len(shape[0])):
            row.append(shape[r][len(shape[r]) - c - 1])
        data.append(row)
    return data


def rotate(shape, degrees):
    if degrees not in (90, 180, 270):
        raise Exception()

    def rotate90(obj):
        data = []
        for c in range(len(shape[0])):
            row = []
            for r in range(len(shape))[::-1]:
                row.append(shape[r][c])
            data.append(row)
        return data

    shp = copy.copy(shape)
    if degrees >= 90:
        shp = rotate90(shape)
    if degrees >= 180:
        shp = rotate90(shape)
    if degrees >= 270:
        shp = rotate90(shape)
    return shp


def print_shape(shape):
    shape = list(shape)
    printerr(len(shape), len(shape[0]), empty_space(shape))
    for r in range(len(shape)):
        row = ""
        for c in range(len(shape[0])):
            row += "█" if shape[r][c] else "░"
        printerr(row)


def merge_shapes_vertical(a, b, divide=0, divider=False):
    assert len(a) == len(b)

    out = []

    for r in range(len(a)):
        row = []
        for c in range(len(a[r])):
            row.append(a[r][c])

        if divide:
            if divider:
                if row[-1]:
                    row.extend([False, True, False])
                else:
                    row.extend([False, False, False])
            else:
                for i in range(int(divide)):
                    row.extend([False])
        else:
            pass

        for c in range(len(b[r])):
            row.append(b[r][c])

        out.append(row)
    return out


def merge_shapes_horizontal(a, b, divide=False):
    out = []

    for r in range(len(a)):
        row = []
        for c in range(len(a[r])):
            row.append(a[r][c])
        out.append(row)

    if divide:
        out.append([False] * len(a[0]))

    for r in range(len(b)):
        row = []
        for c in range(len(b[r])):
            row.append(b[r][c])
        out.append(row)

    return out


def gen_petal(petal_width, elongation, pointy=False):
    temp_shape = []

    for r in range(petal_width):
        row = []
        for c in range(petal_width):
            row.append(c <= r)
        temp_shape.append(row)

    for r in range(elongation):
        row = []
        for c in range(petal_width):
            row.append(True)
        temp_shape.append(row)

    for r in range(petal_width):
        row = []
        for c in range(petal_width):
            row.append(c > r)
        temp_shape.append(row)

    if not pointy:
        temp_shape[0][0] = False
        temp_shape[1][0] = False

    return temp_shape


def gen_flower(petal_width, elongation, pointy=False, divide=False):
    temp_shape = gen_petal(petal_width, elongation, pointy=pointy)

    temp_shape_mirror = mirror_vertical(temp_shape)
    combined = merge_shapes_vertical(
        temp_shape, temp_shape_mirror, divide=divide
    )

    return combined


class Scarf:
    def __init__(self, width=51, length=500, pixel=15):
        self.length = length
        self.width = width
        self.pixel = pixel

        self.pieces = [End(width=self.width, height=15)]
        self.type = 'ends' if random.random() > 0.8 else 'complete'

        pbar = tqdm.tqdm(total=self.length)
        while True:
            # if previous piece was a border or an end, generate a block.
            prev_piece = self.pieces[-1]
            if isinstance(prev_piece, (End, Border)):
                b = Block(width=width, height=random.choice([15, 25, 35]))
                if random.random() < 0.3:
                    b.invert()


            else:
                # otherwise it was a block so we should generate a border piece
                b = Border(width=width, height=random.choice([10, 14]))

            if not b.acceptable:
                continue

            if not b.fits_below(prev_piece):
                continue

            self.pieces.append(b)

            pbar.update(b.height)
            if sum([x.height for x in self.pieces]) > length - 15:
                break
        pbar.close()
        self.pieces.append(End(width=self.width, height=15))


    def render(self, handle=sys.stdout):
        handle.write(
            svg.header(
                "Helena",
                "scarf",
                width=self.width * self.pixel,
                height=self.length * self.pixel,
            )
        )

        height_offset = 0
        for block in self.pieces:
            handle.write(svg.group_begin())
            for r in range(block.height):
                for c in range(self.width):
                    props = {
                        "fill": "red" if block.data[r][c] else "white",
                        "opacity": 1,
                        "stroke": "none",
                    }
                    handle.write(
                        svg.rect(
                            x=c * self.pixel,
                            y=(height_offset + r) * self.pixel,
                            width=self.pixel,
                            height=self.pixel,
                            style=props,
                        )
                    )
            height_offset += block.height
            handle.write(svg.group_end())

        handle.write(svg.footer())


class Block:
    def __init__(self, width=51, height=15, pixel=25):
        self.width = width
        self.height = height
        self.pixel = pixel
        self.acceptable = True

        self.data = empty(self.height, self.width)
        self.generate()

    def render(self):
        self.header()
        self.print()
        self.footer()

    def generate(self):
        self._generate()
        self.calc_metadata()

    def fits_below(self, other):
        return self.border_top_color == other.border_bot_color

    def _generate(self):

        # Pick a repeat length
        # repeat = random.choice([10, 20, 30])
        # Given the height we'll fit something in here.
        height = self.height
        # symmetry is important!

        base_shape = random.choice(["8flower", "braid", "small_flower", "flipping_hearts"])  # , 'flower', 'square', '4x'])
        # base_shape = "flipping_hearts"


        center_x = self.width // 2
        center_y = self.height // 2

        if base_shape == "8flower":
            petal_sep_cardinal = random.random() > 0.3
            petal_height = int(0.8 * height / 2) * 2
            if petal_sep_cardinal:
                petal_height -= 3

            # petal_padding = (height - petal_height) / 2
            elongation = random.choice([0, 1, 2, 3])
            # so how much space we have is dependent on shape height / 2
            max_petal_size = (petal_height / 2) - 3
            if petal_sep_cardinal:
                max_petal_size -= 2

            pointy = random.random() > 0.5

            # Now we're ready to plot. We know if we will separate the petals,
            # and we know how long to elongate them. Maybe start in nearly the
            # center of the block?

            temp_shape = []
            # plot petal nnw
            # 2x width + elongation
            # so width:
            petal_width = max(4, int((max_petal_size - elongation) // 2))

            # cap random choice earlier?
            if 2 * (petal_width * 2 + elongation) > self.height:
                elongation = int((self.height / 2) - (petal_width * 2))

            for r in range(petal_width):
                row = []
                for c in range(petal_width):
                    row.append(c <= r)
                temp_shape.append(row)

            for r in range(elongation):
                row = []
                for c in range(petal_width):
                    row.append(True)
                temp_shape.append(row)

            for r in range(petal_width):
                row = []
                for c in range(petal_width):
                    row.append(c > r)
                temp_shape.append(row)

            if not pointy:
                temp_shape[0][0] = False
                temp_shape[1][0] = False

            temp_shape_mirror = mirror_vertical(temp_shape)
            combined = merge_shapes_vertical(
                temp_shape, temp_shape_mirror, divide=petal_sep_cardinal
            )
            flip_vert = mirror_horizontal(combined)
            combined2 = merge_shapes_horizontal(
                combined, flip_vert, divide=petal_sep_cardinal
            )


            blit(
                self.data,
                combined2,
                x=center_x - petal_width,
                y=(self.height - len(combined2)) // 2,
            )

            rot90 = rotate(combined2, degrees=90)

            blit(
                self.data,
                rot90,
                x=int(center_x - (len(rot90[0]) / 2))
                + (1 if petal_sep_cardinal else 0),
                y=(self.height - len(rot90)) // 2,
            )

            sym = symmetry_horiz(self.data)
            self.acceptable = sym >= 0.95
        elif base_shape == "braid":

            # We want the biggest number <= braid_height
            # braid_height_max = 3 * height
            braid_height = (8 * self.height) // 10
            if braid_height < 4:
                return

            width = random.choice([1, 2, 3, 4, 5])

            block_unit = empty(braid_height, braid_height - width + 1)

            for x in range(braid_height):
                for y in range(braid_height - width + 1):
                    if x == y:
                        for i in range(width):
                            gs(block_unit, x + i, y, True)

            inv = mirror_horizontal(block_unit)

            mask = eat(block_unit, 1, diag=True)

            a = list(subtract(inv, mask))
            b = list(overlay(a, block_unit))

            while True:
                b = merge_shapes_vertical(b, b)
                if len(b[0]) > self.width:
                    break

            b = trim(b, width=self.width)

            blit(self.data, b, y=center_y - (braid_height//2))
        elif base_shape == "small_flower":
            petal_sep_cardinal = random.random() > 0.3
            petal_height = int(0.8 * height / 2)
            if petal_sep_cardinal:
                petal_height -= 3

            # petal_padding = (height - petal_height) / 2
            elongation = random.choice([0, 1, 2, 3])
            # so how much space we have is dependent on shape height / 2
            max_petal_size = (petal_height / 2) - 3
            if petal_sep_cardinal:
                max_petal_size -= 2

            pointy = random.random() > 0.5
            petal_width = max(4, int((max_petal_size - elongation) // 2))
            # cap it
            if 2 * (petal_width * 2 + elongation) > self.height:
                elongation = int((self.height / 2) - (petal_width * 2))

            flower = gen_flower(petal_width, elongation, pointy=pointy, divide=petal_sep_cardinal)
            flip_vert = mirror_horizontal(flower)
            combined2 = merge_shapes_horizontal(
                flower, flip_vert, divide=petal_sep_cardinal
            )

            space = random.choice([1, 2, 3, 4])
            newsize = space + len(combined2[0])
            diff = divmod(self.width, newsize)[1]
            # This is a hack around centering since ugh math
            xoff = random.choice([1, 2, 3, 4])
            for i in range(1 + (self.width // newsize)):
                blit(
                    self.data,
                    combined2,
                    x=(i * newsize) + xoff,
                    y=(self.height - len(combined2)) // 2,
                )
            sym = symmetry_horiz(self.data)
            self.acceptable = sym >= 0.95
        elif base_shape == "flipping_hearts":
            petal_height = int(0.8 * height / 2)

            elongation = random.choice([0, 1, 2, 3])
            # so how much space we have is dependent on shape height / 2
            max_petal_size = (petal_height / 2) - 3

            petal_width = max(4, int((max_petal_size - elongation) // 2))
            # cap it
            if 2 * (petal_width * 2 + elongation) > self.height:
                elongation = int((self.height / 2) - (petal_width * 2))

            flower = gen_flower(petal_width, elongation, pointy=False, divide=False)
            flip_vert = mirror_horizontal(flower)
            combined2 = merge_shapes_vertical(flower, flip_vert, divide=2)

            while True:
                combined2 = merge_shapes_vertical(combined2, combined2, divide=2)
                if len(combined2[0]) > self.width:
                    break

            blit(
                self.data,
                combined2,
                x=random.choice([0, 1, 2, 3, 4]),
                y=(self.height - len(combined2)) // 2,
            )
            sym = symmetry_horiz(self.data)
            self.acceptable = sym >= 0.90


        self.fill_space(percentage=random.choice([0.15, 0.25, 0.35]))


    def calc_metadata(self):
        if empty_space(list(trim(self.data, height=2))) < 0.5:
            self.border_top_color = 'light'
        else:
            self.border_top_color = 'dark'

        if empty_space(list(trim(self.data, height=-2))) < 0.5:
            self.border_bot_color = 'light'
        else:
            self.border_bot_color = 'dark'

    def invert(self):
        self.data = list(invert(self.data))
        self.calc_metadata()

    def fill_space(self, percentage=0.15):
        if random.random() > 0.6:
            self._fill_invert(self.data)

            for i in range(10):
                if empty_space(self.data) < percentage:
                    self._fill_space(self.data)
        else:
            self._fill_space(self.data)

            for i in range(10):
                if empty_space(self.data) < percentage:
                    self._fill_space(self.data)

    def _fill_invert(self, data):
        if random.random() > 0.3:
            mask = eat(self.data, 2, diag=True)
            mask = list(invert(mask))
            self.data = list(overlay(mask, data))
        else:
            mask1 = copy.copy(self.data)
            for i in range(1, 5):
                mask_tmp = eat(self.data, i * 2, diag=True)
                mask1 = list(xor(mask1, mask_tmp))

            self.data = list(overlay(mask1, data))

            m = empty_space(self.data)
            if m > 0.5:
                self.invert()


    def _fill_space(self, data):
        method = random.choice(["dots_rect", "cross", "cempty"])

        if method == "dots_rect":
            shape = [[True]]
        elif method == "cross":
            shape = [[False, True, False], [True, True, True], [False, True, False]]
        elif method == "cempty":
            shape = [[False, True, False], [True, False, True], [False, True, False]]

        v_div = random.choice([x for x in divisors(self.height - len(shape)) if x >= 3])
        h_div = random.choice([x for x in divisors(self.width - len(shape)) if x >= 3])

        for r in range(len(data)):
            for c in range(len(data[0])):
                if r % v_div == 0 and c % h_div == 0:
                    blit(data, shape, x=c, y=r, gap=True)


    def print(self):
        for r in range(self.height):
            for c in range(self.width):
                props = {
                    "fill": "red" if self.data[r][c] else "white",
                    "opacity": 1,
                    "stroke": "none",
                }
                print(
                    svg.rect(
                        c * self.pixel,
                        r * self.pixel,
                        self.pixel,
                        self.pixel,
                        style=props,
                    )
                )

    def footer(self):
        print(svg.footer())

    def header(self):
        print(
            svg.header(
                "Helena",
                "scarf",
                width=self.width * self.pixel,
                height=self.height * self.pixel,
            )
        )


class Border(Block):

    def _generate(self):
        type = random.choice(['fadeUp', 'fadeDown', 'check2', 'check3'])

        if type == 'fadeUp':
            for r in range(len(self.data)):
                for c in range(len(self.data[r])):
                    self.data[r][c] = random.random() > r / len(self.data)
        elif type == 'fadeDown':
            for r in range(len(self.data)):
                for c in range(len(self.data[r])):
                    self.data[r][c] = random.random() < r / len(self.data)
        elif type == 'plain':
            for r in range(len(self.data)):
                for c in range(len(self.data[r])):
                    self.data[r][c] = r <= 1 or r >= len(self.data) - 2
        elif type == 'check':
            for r in range(len(self.data)):
                for c in range(len(self.data[r])):
                    self.data[r][c] = (r + c) % 2 == 0
        elif type == 'check2':
            for r in range(len(self.data)):
                for c in range(len(self.data[r])):
                    self.data[r][c] = ((r + c) // 2) % 2 == 0
        elif type == 'check3':
            for r in range(len(self.data)):
                for c in range(len(self.data[r])):
                    self.data[r][c] = ((r + c) // 2) % 3 == 0

            q = list(mirror_horizontal(self.data))
            self.data = list(xor(self.data, q))

            for c in range(len(self.data[0])):
                self.data[0][c] = True
                self.data[len(self.data) - 1][c] = True

        elif type == 'stripes_h':
            width = random.choice([1, 2, 3, 4, 5])
            for r in range(len(self.data)):
                for c in range(len(self.data[r])):
                    self.data[r][c] = (c // width) % 2 == 0
        elif type == 'stripes_h':
            width = random.choice([1, 2, 3, 4, 5])
            for r in range(len(self.data)):
                for c in range(len(self.data[r])):
                    self.data[r][c] = (c // width) % 2 == 0


class End(Block):
    def _generate(self):
        self.data = empty(self.height, self.width, True)

    def fits_below(self, other):
        return True


# q = Block()
s = Scarf()
s.render()
