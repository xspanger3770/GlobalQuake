package globalquake.ui.table;

import javax.swing.*;
import java.awt.*;

public class ProgressBarRenderer<E> extends TableCellRendererAdapter<E, JProgressBar> {
    @Override
    public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int column) {
        return (JProgressBar) value;
    }
}
