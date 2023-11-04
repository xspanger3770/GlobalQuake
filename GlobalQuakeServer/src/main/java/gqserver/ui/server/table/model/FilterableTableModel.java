package gqserver.ui.server.table.model;

import gqserver.ui.server.table.TableCellRendererAdapter;

import java.io.Serial;
import java.util.ArrayList;
import java.util.List;

import javax.swing.table.AbstractTableModel;

@SuppressWarnings("unused")
public abstract class FilterableTableModel<E> extends AbstractTableModel {

	@Serial
	private static final long serialVersionUID = 1727941556193013022L;
	private final List<E> data;
	private final List<E> filteredData;

	public FilterableTableModel(List<E> data) {
		this.data = data;
		this.filteredData = new ArrayList<>(data);
		applyFilter();
	}

	public synchronized final void applyFilter() {
		this.filteredData.clear();
		this.filteredData.addAll(this.data.stream().filter(this::accept).toList());
		super.fireTableDataChanged();
	}

	@SuppressWarnings("SameReturnValue")
	public boolean accept(E entity) {
		return true;
	}

	@SuppressWarnings("EmptyMethod")
	public void dataUpdated() {

	}

	@Override
	public synchronized int getRowCount() {
		return filteredData.size();
	}

	public synchronized void updateRow(E entity) {
		int rowIndex;
		rowIndex = filteredData.indexOf(entity);
		fireTableRowsUpdated(rowIndex, rowIndex);
		dataUpdated();
	}

	public synchronized void deleteRow(int rowIndex) {
		E entity;
		entity = filteredData.get(rowIndex);
		filteredData.remove(entity);
		data.remove(entity);
		fireTableRowsDeleted(rowIndex, rowIndex);
		dataUpdated();
	}

	public synchronized void addRow(E entity) {
		int newRowIndex;
		newRowIndex = filteredData.size();
		filteredData.add(entity);

		data.add(entity);
		fireTableRowsInserted(newRowIndex, newRowIndex);
		dataUpdated();
	}

	public synchronized E getEntity(int rowIndex) {
		return filteredData.get(rowIndex);
	}

	public List<E> getData() {
		return data;
	}

	public abstract TableCellRendererAdapter<?, ?> getColumnRenderer(int columnIndex);
}
