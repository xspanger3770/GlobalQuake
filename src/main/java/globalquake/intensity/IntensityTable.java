package globalquake.intensity;

public class IntensityTable {

    private static final int ROWS = 100; // UP TO 100,000km
    private static final int COLS = 120; // M-2 - M10
    private static final double[][] TABLE;
    private static final double[][] TABLE2;

    static {
        TABLE = new double[ROWS][COLS];
        TABLE2 = new double[ROWS][COLS];
        fillTables();
    }

    private static double getDist(double row) {
        return 10.0 * (Math.pow(11.0 / 10.0, row) - 1);
    }

    private static double getRow(double dist) {
        return Math.log10(dist / 10.0 + 1) / Math.log10(11.0 / 10.0);
    }

    private static double getMag(double col) {
        return -2.0 + col / 10.0;
    }

    private static double getCol(double mag) {
        return (mag + 2.0) * 10;
    }

    private static double getIntensity(double col) {
        return 10.0 * (Math.pow(11.5 / 10.0, col) - 1);
    }

    private static double getColByIntensity(double intensity) {
        return Math.log10(intensity / 10.0 + 1) / Math.log10(11.5 / 10.0);
    }

    private static void fillTables() {
        long a = System.currentTimeMillis();
        for (int row = 0; row < ROWS; row++) {
            for (int col = 0; col < COLS; col++) {
                double dist = getDist(row);
                double mag = getMag(col);
                TABLE[row][col] = maxIntensity(mag, dist);
            }
        }
        for (int row = 0; row < ROWS; row++) {
            for (int col = 0; col < COLS; col++) {
                double intensity = getIntensity(col);
                double mag = searchMag(row, intensity);
                TABLE2[row][col] = mag;
            }
        }
        System.out.printf("Intensity tables filled in %s ms.\n", (System.currentTimeMillis() - a));
    }

    private static double searchMag(int row, double intensity) {
        // TOO LAZY FOR BINARY SEARCH, WILL BE USED ONLY AT INIT
        for (int col = 0; col < COLS; col++) {
            double mag = getMag(col);
            double _intensity = TABLE[row][col];
            if (_intensity >= intensity) {
                return mag;
            }
        }
        return -2;
    }

    /**
     * @param mag  Magnitude
     * @param dist 'Geological' distance in KM
     * @return Expected intensity
     */

    public static double getMaxIntensity(double mag, double dist) {
        double _row = getRow(dist);
        double _col = getCol(mag);
        return extrapolateVal(_row, _col, TABLE);
    }

    private static double extrapolateVal(double _row, double _col, double[][] table) {
        int row0 = (int) (Math.max(0, Math.min(ROWS - 2, _row)));
        int col0 = (int) (Math.max(0, Math.min(COLS - 2, _col)));
        double valAB = table[row0][col0] * (1 - _col % 1.0) + table[row0][col0 + 1] * (_col % 1.0);
        double valCD = table[row0 + 1][col0] * (1 - _col % 1.0) + table[row0 + 1][col0 + 1] * (_col % 1.0);
        return valAB * (1 - _row % 1.0) + valCD * (_row % 1.0);
    }

    /**
     * @param dist      'Geological' distance in KM
     * @param intensity maxRatio
     * @return Magnitude
     */

    public static double getMagnitude(double dist, double intensity) {
        double _row = getRow(dist);
        double _col = getColByIntensity(intensity);
        return extrapolateVal(_row, _col, TABLE2);
    }

    private static double maxIntensity(double mag, double dist) {
        mag = 1.2 * mag - 0.022 * mag * mag - 1;
        if (dist > 1200) {
            dist = 1200 + Math.pow(dist - 1200, 0.4) * 22.0;
        }
        return (Math.pow(15, mag * 0.92 + 4.0)) / (5 * Math.pow(dist, 2.1 + 0.07 * mag) + 1 * Math.pow(5, mag));

    }

    public static void init() {
        getMaxIntensity(0,0);
    }
}
