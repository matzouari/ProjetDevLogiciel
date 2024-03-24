import type { Transform } from "./base";
import type { LineGLVisuals } from "./base_line";
import { BaseLineGL } from "./base_line";
import { Uint8Buffer } from "./buffer";
import type { ReglWrapper } from "./regl_wrap";
import type { LineView } from "../line";
export declare class LineGL extends BaseLineGL {
    readonly glyph: LineView;
    constructor(regl_wrapper: ReglWrapper, glyph: LineView);
    draw(indices: number[], main_glyph: LineView, transform: Transform): void;
    protected _get_show_buffer(indices: number[], main_gl_glyph: LineGL): Uint8Buffer;
    protected _get_visuals(): LineGLVisuals;
    protected _set_data_points(): Float32Array;
}
//# sourceMappingURL=line_gl.d.ts.map