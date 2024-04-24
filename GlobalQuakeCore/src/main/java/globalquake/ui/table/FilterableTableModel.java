package globalquake.ui.table;

import java.io.Serial;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import javax.swing.table.AbstractTableModel;

public abstract class FilterableTableModel<E> extends AbstractTableModel {

    @Serial
    private static final long serialVersionUID = 1727941556193013022L;
    private final Collection<E> data;
    private final List<E> filteredData;

    public FilterableTableModel(Collection<E> data) {
        this.data = data;
        this.filteredData = new ArrayList<>(data);
        applyFilter();
    }

    public synchronized final void applyFilter() {
        this.filteredData.clear();
        this.filteredData.addAll(this.data.stream().filter(this::accept).toList());
        super.fireTableDataChanged();
    }

    @SuppressWarnings({"unused", "SameReturnValue"})
    public boolean accept(E entity) {
        return true;
    }

    public synchronized E getEntity(int rowIndex) {
        return filteredData.get(rowIndex);
    }

    public abstract TableCellRendererAdapter<?, ?> getColumnRenderer(int columnIndex);

    @Override
    public synchronized int getRowCount() {
        return filteredData.size();
    }
}
