package globalquake.core.intensity;

import java.util.List;

public interface IntensityScale {

    /**
     * @return List of levels sorted from biggest to smallest
     */
    List<Level> getLevels();

    String getNameShort();

    @SuppressWarnings("unused")
    String getNameLong();

    double getDarkeningFactor();

    default Level getLevel(double pga) {
        Level result = null;
        for (Level level : getLevels()) {
            if (pga < level.getPga()) {
                return result;
            }

            result = level;
        }

        return result;
    }

}
