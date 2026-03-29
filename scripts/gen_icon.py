"""Gera o icone do projeto: retangulos geometricos sobrepostos em fundo grafite escuro."""

from PIL import Image, ImageDraw


def rounded_rect(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, int, int, int],
) -> None:
    """Desenha retangulo com cantos arredondados."""
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill)


def generate_icon(size: int = 1024) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Fundo: grafite escuro com cantos arredondados
    bg_radius = size // 5
    draw.rounded_rectangle(
        (0, 0, size - 1, size - 1),
        radius=bg_radius,
        fill=(30, 30, 36, 255),
    )

    # Margem interna
    margin = size // 7
    area = size - 2 * margin

    # Tres retangulos sobrepostos representando layers de UI/janelas
    # Cada um ligeiramente deslocado, criando profundidade

    layers = [
        # (offset_x, offset_y, width_ratio, height_ratio, color_rgba)
        # Layer de fundo - azul escuro
        (0.0, 0.0, 0.75, 0.70, (45, 85, 140, 180)),
        # Layer do meio - teal
        (0.15, 0.12, 0.75, 0.70, (50, 140, 160, 200)),
        # Layer da frente - azul claro vibrante
        (0.30, 0.24, 0.75, 0.70, (80, 170, 220, 230)),
    ]

    layer_radius = size // 18

    for ox_ratio, oy_ratio, w_ratio, h_ratio, color in layers:
        x0 = margin + int(area * ox_ratio)
        y0 = margin + int(area * oy_ratio)
        x1 = x0 + int(area * w_ratio)
        y1 = y0 + int(area * h_ratio)

        rounded_rect(draw, (x0, y0, x1, y1), layer_radius, color)

        # Linhas internas simulando conteudo de UI (barras horizontais)
        bar_margin_x = int(area * 0.06)
        bar_height = max(size // 60, 3)
        bar_gap = int(area * 0.07)
        bar_y_start = y0 + int(area * 0.12)

        bar_color = (255, 255, 255, 50)

        for i in range(3):
            by = bar_y_start + i * bar_gap
            if by + bar_height > y1 - bar_margin_x:
                break
            # Barras com larguras decrescentes para parecer texto/conteudo
            bar_width_ratio = [0.7, 0.5, 0.35][i]
            bx1 = x0 + bar_margin_x
            bx2 = bx1 + int((x1 - x0 - 2 * bar_margin_x) * bar_width_ratio)
            draw.rounded_rectangle(
                (bx1, by, bx2, by + bar_height),
                radius=bar_height // 2,
                fill=bar_color,
            )

    return img


def main() -> None:
    icon = generate_icon(1024)

    # Salvar em multiplos tamanhos
    output_dir = "assets"
    import os

    os.makedirs(output_dir, exist_ok=True)

    # PNG principal 256x256 (usado pelo winit)
    icon_256 = icon.resize((256, 256), Image.LANCZOS)
    icon_256.save(f"{output_dir}/icon_256x256.png")

    # ICO com multiplos tamanhos para Windows
    icon.save(
        f"{output_dir}/icon.ico",
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )

    # PNG 1024 para referencia/marketing
    icon.save(f"{output_dir}/icon_1024x1024.png")

    print(f"Icones gerados em {output_dir}/")
    print(f"  icon_256x256.png   - para winit (janela)")
    print(f"  icon.ico           - para Windows (multi-size)")
    print(f"  icon_1024x1024.png - referencia alta resolucao")


if __name__ == "__main__":
    main()
