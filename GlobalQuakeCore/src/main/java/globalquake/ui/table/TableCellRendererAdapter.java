package globalquake.ui.table;


import globalquake.core.GlobalQuake;

import javax.swing.*;
import javax.swing.table.DefaultTableCellRenderer;
import java.awt.*;

@SuppressWarnings("unused")
public class TableCellRendererAdapter<E, T> extends DefaultTableCellRenderer {

    public TableCellRendererAdapter() {
        setHorizontalAlignment(SwingConstants.CENTER);
    }

    public TableCellRendererAdapter(int aligment) {
        setHorizontalAlignment(aligment);
    }

    @SuppressWarnings("unchecked")
    @Override
    public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus,
                                                   int row, int column) {
        super.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, column);

        try {
            if (table.getModel() instanceof FilterableTableModel<?>) {
                E entity = (E) ((FilterableTableModel<?>) table.getModel())
                        .getEntity(table.getRowSorter().convertRowIndexToModel(row));
                T t = (T) value;
                Color bck = getBackground(entity, t);
                if (bck != null) {
                    setBackground(bck);
                }
                //sets the background color if selected or not
                //noinspection ConditionalExpressionWithIdenticalBranches
                setForeground(isSelected ? Color.BLACK : Color.BLACK);
                setText(getText(entity, t));
            }

        } catch (ClassCastException e) {
            GlobalQuake.getErrorHandler().handleException(e);
        }

        return this;
    }

    public String getText(E entity, T value) {
        return getText();
    }

    public Color getForeground(E entity, T value) {
        return getForeground();
    }

    @SuppressWarnings("SameReturnValue")
    public Color getBackground(E entity, T value) {
        return null;
    }

}
