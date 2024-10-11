# %%
import numpy as np
from enum import Enum
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import List, Tuple

# %%
class LightnessLevel(Enum):
    MIN = 15
    MAX_DEFAULT = 95


class MonochromePaletteGenerator:
    def __init__(
        self, max_palette_lightness: int = LightnessLevel.MAX_DEFAULT.value
    ) -> None:
        """
        Initialize the MonochromePaletteGenerator with a maximum palette lightness.

        Args:
            max_palette_lightness (int): Maximum lightness value allowed in the palette. Default is 95.

        Example:
            palette_generator = MonochromePaletteGenerator()
        """
        self.max_palette_lightness: int = max_palette_lightness

    def __validate_starting_lightness(self, starting_lightness: int) -> bool:
        """
        Validate if the starting lightness is less than or equal to the minimum allowed lightness.

        Args:
            starting_lightness (int): The lightness value of the starting color.

        Returns:
            bool: True if starting lightness is valid, else False.

        Example:
            palette_generator.__validate_starting_lightness(20)
        """
        return starting_lightness <= LightnessLevel.MIN.value

    @staticmethod
    def hex_to_hsl(hex_color: str) -> Tuple[int, int, int]:
        """
        Convert a HEX color code to HSL format.

        Args:
            hex_color (str): A HEX color string (e.g., '#FFFFFF').

        Returns:
            Tuple[int, int, int]: A tuple containing the Hue, Saturation, and Lightness (H, S, L).

        Example:
            MonochromePaletteGenerator.hex_to_hsl("#FFFFFF")
        """
        hex_color = hex_color.lstrip("#")
        red, green, blue = [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]

        max_rgb_value = max(red, green, blue)
        min_rgb_value = min(red, green, blue)
        color_diff = max_rgb_value - min_rgb_value
        lightness = (max_rgb_value + min_rgb_value) / 2

        if color_diff == 0:
            hue = 0
            saturation = 0
        else:
            saturation = (
                color_diff / (2.0 - max_rgb_value - min_rgb_value)
                if lightness >= 0.5
                else color_diff / (max_rgb_value + min_rgb_value)
            )
            if max_rgb_value == red:
                hue = ((green - blue) / color_diff) % 6
            elif max_rgb_value == green:
                hue = (blue - red) / color_diff + 2
            else:
                hue = (red - green) / color_diff + 4
            hue *= 60
            if hue < 0:
                hue += 360

        return (
            int(round(hue)),
            int(round(saturation * 100)),
            int(round(lightness * 100)),
        )

    @staticmethod
    def hsl_to_hex(hue: int, saturation: int, lightness: int) -> str:
        """
        Convert an HSL color code to HEX format.

        Args:
            hue (int): Hue value (0 to 360 degrees).
            saturation (int): Saturation percentage (0 to 100).
            lightness (int): Lightness percentage (0 to 100).

        Returns:
            str: A HEX color string (e.g., '#FFFFFF').

        Example:
            MonochromePaletteGenerator.hsl_to_hex(200, 100, 50)
        """
        saturation /= 100
        lightness /= 100

        chroma = (1 - abs(2 * lightness - 1)) * saturation
        secondary_component = chroma * (1 - abs((hue / 60) % 2 - 1))
        match_lightness = lightness - chroma / 2

        if 0 <= hue < 60:
            red_prime, green_prime, blue_prime = chroma, secondary_component, 0
        elif 60 <= hue < 120:
            red_prime, green_prime, blue_prime = secondary_component, chroma, 0
        elif 120 <= hue < 180:
            red_prime, green_prime, blue_prime = 0, chroma, secondary_component
        elif 180 <= hue < 240:
            red_prime, green_prime, blue_prime = 0, secondary_component, chroma
        elif 240 <= hue < 300:
            red_prime, green_prime, blue_prime = secondary_component, 0, chroma
        elif 300 <= hue < 360:
            red_prime, green_prime, blue_prime = chroma, 0, secondary_component
        else:
            raise ValueError("Hue value must be in range [0, 360)")

        red = round((red_prime + match_lightness) * 255)
        green = round((green_prime + match_lightness) * 255)
        blue = round((blue_prime + match_lightness) * 255)

        return f"#{red:02X}{green:02X}{blue:02X}"

    def __create_lightness_range(
        self, starting_lightness: int, num_colors: int
    ) -> np.ndarray:
        """
        Create a list of linearly spaced lightness values for the palette.

        Args:
            starting_lightness (int): The lightness value of the starting color.
            num_colors (int): The number of colors to generate.

        Returns:
            np.ndarray: An array of lightness values.

        Example:
            palette_generator.__create_lightness_range(20, 5)
        """
        lightness_values = np.linspace(
            starting_lightness, self.max_palette_lightness, num_colors
        )
        return np.round(lightness_values).astype(int)

    def create_hex_code_palette(self, starting_hex: str, num_colors: int) -> List[str]:
        """
        Generate a monochrome palette in HEX format.

        Args:
            starting_hex (str): The starting color in HEX format.
            num_colors (int): The number of colors to generate.

        Returns:
            List[str]: A list of HEX color codes forming the palette.

        Raises:
            ValueError: If the starting color is too light.

        Example:
            palette_generator.create_hex_code_palette("#051923", 10)
        """
        starting_hsl = self.hex_to_hsl(hex_color=starting_hex)
        if self.__validate_starting_lightness(starting_lightness=starting_hsl[2]):
            lightness_range = self.__create_lightness_range(
                starting_lightness=starting_hsl[2], num_colors=num_colors
            )
            return [
                self.hsl_to_hex(
                    hue=starting_hsl[0], saturation=starting_hsl[1], lightness=lightness
                )
                for lightness in lightness_range
            ]
        raise ValueError(
            "Given starting color is too light to construct a palette. Please choose a darker shade."
        )

    @staticmethod
    def create_matplotlib_palette(
        colors: List[str], palette_name: str
    ) -> mpl.colors.ListedColormap:
        """
        Create a matplotlib colormap from a list of colors.

        Args:
            colors (List[str]): A list of HEX color codes.
            palette_name (str): The name for the colormap.

        Returns:
            mpl.colors.ListedColormap: A matplotlib ListedColormap object.

        Example:
            palette_generator.create_matplotlib_palette(["#051923", "#102A3C"], "custom_palette")
        """
        return mpl.colors.ListedColormap(colors, name=palette_name)


# %%
palette_generator = MonochromePaletteGenerator()

# %%
monochrome_blue = palette_generator.create_hex_code_palette(
    starting_hex="#051923", num_colors=10
)
monochrome_blue_mpl = palette_generator.create_matplotlib_palette(
    colors=monochrome_blue, palette_name="custom-monochrome-blue"
)
print(monochrome_blue)
monochrome_blue_mpl

# %%
monochrome_red = palette_generator.create_hex_code_palette(
    starting_hex="#1c0000", num_colors=7
)
monochrome_red_mpl = palette_generator.create_matplotlib_palette(
    colors=monochrome_red, palette_name="custom-monochrome-red"
)
print(monochrome_red)
monochrome_red_mpl

# %%
df = pd.DataFrame(
    {
        "HR": [50, 63, 40, 68, 35],
        "Engineering": [77, 85, 62, 89, 58],
        "Marketing": [50, 35, 79, 43, 67],
        "Sales": [59, 62, 33, 77, 72],
        "Customer Service": [31, 34, 61, 70, 39],
        "Distribution": [35, 21, 66, 90, 31],
        "Logistics": [50, 54, 13, 71, 32],
        "Production": [22, 51, 54, 28, 40],
        "Maintenance": [50, 32, 61, 69, 50],
        "Quality Control": [20, 21, 88, 89, 39],
    },
    index=["New York", "San Francisco", "Los Angeles", "Chicago", "Miami"],
)

df = df.T
df = df.loc[df.sum(axis=1).sort_values().index]

df

# %%
ax = df.plot(
    kind="barh",
    colormap=monochrome_blue_mpl,
    width=0.8,
    edgecolor="#000000",
    stacked=True,
)

plt.title(
    "Employee Count Per Location And Department",
    loc="left",
    fontdict={"weight": "bold"},
    y=1.06,
)
plt.xlabel("Count")
plt.ylabel("Office Location")
plt.show()

# %%
ax = df.plot(
    kind="barh",
    colormap=monochrome_red_mpl,
    width=0.8,
    edgecolor="#000000",
    stacked=True,
)

plt.title(
    "Employee Count Per Location And Department",
    loc="left",
    fontdict={"weight": "bold"},
    y=1.06,
)
plt.xlabel("Count")
plt.ylabel("Office Location")
plt.show()
